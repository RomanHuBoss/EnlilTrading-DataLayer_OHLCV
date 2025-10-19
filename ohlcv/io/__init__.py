# ohlcv/io/__init__.py
from __future__ import annotations

# Упрощённые импорты IO-уровня
from .parquet_store import parquet_path, write_idempotent  # noqa: F401
from . import tail_cache as tail  # noqa: F401

__all__ = [
    "parquet_path",
    "write_idempotent",
    "tail",
]
