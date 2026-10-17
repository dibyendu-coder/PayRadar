"""Inspect installed tabpfn package metadata, docstrings and signatures."""

import inspect

import tabpfn
from tabpfn import TabPFNClassifier, TabPFNRegressor

print("=" * 60)
print(f"TabPFN Version: {getattr(tabpfn, '__version__', 'unknown')}")
print(f"TabPFN Package Path: {tabpfn.__file__}")
print("=" * 60)

print("\n--- TabPFNClassifier ---")
print("Init Signature:", inspect.signature(TabPFNClassifier.__init__))
print("Docstring Summary:\n", (TabPFNClassifier.__doc__ or "")[:500])

print("\n--- TabPFNRegressor ---")
print("Init Signature:", inspect.signature(TabPFNRegressor.__init__))
print("Docstring Summary:\n", (TabPFNRegressor.__doc__ or "")[:500])
