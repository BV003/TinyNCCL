"""Minimal assertion helper that is not stripped under ``python -O``."""
from __future__ import annotations


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
