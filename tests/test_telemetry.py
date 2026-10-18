"""Unit tests for Sentry Agent Tracing and PII scrubbing in payradar.telemetry."""

import pytest

from payradar.telemetry import (
    GemmaCallTracker,
    init_telemetry,
    is_telemetry_enabled,
    trace_gemma_call,
)


def test_telemetry_disabled_by_default(monkeypatch):
    """Ensure tracing is inactive when SENTRY_DSN is unset."""
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    assert init_telemetry() is False
    assert is_telemetry_enabled() is False


def test_trace_gemma_call_noop_when_disabled(monkeypatch):
    """Ensure trace_gemma_call operates cleanly without Sentry when disabled."""
    monkeypatch.delenv("SENTRY_DSN", raising=False)

    with trace_gemma_call(operation_name="chase", model_name="gemma4:e2b") as tracker:
        assert isinstance(tracker, GemmaCallTracker)
        tracker.record_response({"prompt_eval_count": 120, "eval_count": 45, "model": "gemma4:e2b"})

    assert tracker.latency_seconds >= 0.0
    assert tracker.input_tokens == 120
    assert tracker.output_tokens == 45
    assert tracker.response_model == "gemma4:e2b"


def test_tracker_records_tokens_and_scrubs_pii():
    """Verify tracker records token metrics while omitting invoice contents or client names."""
    mock_span_data = {}

    class MockSpan:
        def set_data(self, key, value):
            mock_span_data[key] = value

        def set_status(self, status):
            mock_span_data["status"] = status

    span = MockSpan()
    tracker = GemmaCallTracker(span=span)

    # Simulate raw Ollama response
    ollama_resp = {
        "model": "gemma4:e2b",
        "prompt_eval_count": 250,
        "eval_count": 85,
        # Ollama message contains actual text, which should NEVER be copied into span attributes
        "message": {"role": "assistant", "content": "Dear Client, invoice 123 is due."},
    }

    tracker.record_response(ollama_resp)

    # Check that metadata and token metrics were recorded on span
    assert mock_span_data["gen_ai.response.model"] == "gemma4:e2b"
    assert mock_span_data["gen_ai.usage.input_tokens"] == 250
    assert mock_span_data["gen_ai.usage.output_tokens"] == 85
    assert mock_span_data["gen_ai.usage.total_tokens"] == 335

    # Check that NO PII from message content or client details was stored on span
    span_values_str = str(mock_span_data)
    assert "Dear Client" not in span_values_str
    assert "invoice 123" not in span_values_str


def test_trace_gemma_call_records_errors(monkeypatch):
    """Verify that exceptions inside the trace block record error status and type."""
    # Test with mock span
    mock_span_data = {}

    class MockSpan:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def set_data(self, key, value):
            mock_span_data[key] = value

        def set_status(self, status):
            mock_span_data["status"] = status

    # Simulate enabled telemetry
    import sentry_sdk

    monkeypatch.setattr(sentry_sdk, "start_span", lambda **kwargs: MockSpan())
    import payradar.telemetry as tel

    monkeypatch.setattr(tel, "is_telemetry_enabled", lambda: True)

    with pytest.raises(ValueError):
        with trace_gemma_call(operation_name="extract", model_name="gemma4:e2b"):
            raise ValueError("Ollama connection timed out")

    assert mock_span_data["error.type"] == "ValueError"
    assert mock_span_data["status"] == "internal_error"
    assert "gen_ai.latency_seconds" in mock_span_data
