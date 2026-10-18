"""Telemetry and Sentry Agent Tracing instrumentation for PayRadar.

Follows Sentry Agent Tracing specification:
- https://docs.sentry.io/product/agents/getting-started/
- https://docs.sentry.io/platforms/python/agent-tracing/manual-instrumentation/

Non-negotiable requirements:
- SENTRY_DSN comes from environment.
- Tracing is disabled (no-op) if SENTRY_DSN is unset or empty.
- Instrument every Gemma call (extract and chase) with:
  - Latency (seconds)
  - Token counts (gen_ai.usage.input_tokens, gen_ai.usage.output_tokens, gen_ai.usage.total_tokens)
  - Model name (gen_ai.request.model, gen_ai.response.model)
  - Errors (error.type, internal_error status)
- Privacy & scrubbing: NEVER send invoice contents, client names, raw images, or email text to Sentry.
  send_default_pii is strictly False.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Generator

_SENTRY_INITIALIZED = False


def init_telemetry() -> bool:
    """Initialize Sentry SDK if SENTRY_DSN is configured in environment."""
    global _SENTRY_INITIALIZED
    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        _SENTRY_INITIALIZED = False
        return False

    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=dsn,
            send_default_pii=False,
            traces_sample_rate=1.0,
            release="payradar@0.1.0",
            environment=os.environ.get("ENVIRONMENT", "production"),
        )
        _SENTRY_INITIALIZED = True
        return True
    except Exception:
        _SENTRY_INITIALIZED = False
        return False


def is_telemetry_enabled() -> bool:
    """Return whether Sentry tracing is active."""
    global _SENTRY_INITIALIZED
    if not _SENTRY_INITIALIZED:
        init_telemetry()
    return _SENTRY_INITIALIZED


class GemmaCallTracker:
    """Helper to record metrics and token counts on active span without PII."""

    def __init__(self, span: Any | None = None) -> None:
        self.span = span
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.response_model: str | None = None
        self.latency_seconds: float = 0.0

    def record_response(self, ollama_response: dict[str, Any]) -> None:
        """Extract token counts and model name from Ollama API response (never capturing content)."""
        if not isinstance(ollama_response, dict):
            return

        model = ollama_response.get("model")
        if model:
            self.response_model = str(model)
            if self.span is not None:
                self.span.set_data("gen_ai.response.model", self.response_model)

        prompt_eval_count = ollama_response.get("prompt_eval_count")
        eval_count = ollama_response.get("eval_count")

        if prompt_eval_count is not None:
            try:
                self.input_tokens = int(prompt_eval_count)
                if self.span is not None:
                    self.span.set_data("gen_ai.usage.input_tokens", self.input_tokens)
            except (ValueError, TypeError):
                pass

        if eval_count is not None:
            try:
                self.output_tokens = int(eval_count)
                if self.span is not None:
                    self.span.set_data("gen_ai.usage.output_tokens", self.output_tokens)
            except (ValueError, TypeError):
                pass

        if self.input_tokens is not None and self.output_tokens is not None:
            total = self.input_tokens + self.output_tokens
            if self.span is not None:
                self.span.set_data("gen_ai.usage.total_tokens", total)

    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Record manual token counts."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        if self.span is not None:
            self.span.set_data("gen_ai.usage.input_tokens", input_tokens)
            self.span.set_data("gen_ai.usage.output_tokens", output_tokens)
            self.span.set_data("gen_ai.usage.total_tokens", input_tokens + output_tokens)


@contextmanager
def trace_gemma_call(
    operation_name: str,
    model_name: str,
) -> Generator[GemmaCallTracker, None, None]:
    """Context manager instrumenting a local Gemma model call with Sentry Agent Tracing.

    If SENTRY_DSN is unset, this context manager executes locally as a zero-overhead no-op.
    No invoice fields, client names, raw image buffers, or drafted text are recorded or sent.
    """
    start_time = time.perf_counter()

    if not is_telemetry_enabled():
        tracker = GemmaCallTracker(span=None)
        try:
            yield tracker
        finally:
            tracker.latency_seconds = time.perf_counter() - start_time
        return

    import sentry_sdk

    span_attributes = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "ollama",
        "gen_ai.request.model": model_name,
        "gen_ai.agent.name": f"PayRadar-{operation_name}",
        "payradar.operation": operation_name,
    }

    with sentry_sdk.start_span(
        op="gen_ai.chat",
        name=f"chat {model_name}",
        attributes=span_attributes,
    ) as span:
        tracker = GemmaCallTracker(span=span)
        try:
            yield tracker
        except Exception as exc:
            span.set_data("error.type", exc.__class__.__name__)
            try:
                span.set_status("internal_error")
            except Exception:
                pass
            raise
        finally:
            elapsed = time.perf_counter() - start_time
            tracker.latency_seconds = elapsed
            span.set_data("gen_ai.latency_seconds", round(elapsed, 4))
