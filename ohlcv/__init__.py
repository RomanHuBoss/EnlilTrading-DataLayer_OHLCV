"""OHLCV Data Layer (C1)

Публичный API пакета: версия, сигнатура сборки, CLI и основные утилиты.
"""
from __future__ import annotations

from .version import __version__, build_signature

# CLI (c1-ohlcv)
from .cli import main as cli_main  # noqa: F401

# Низкоуровневый клиент Bybit
from .api.bybit import BybitClient, FetchStats  # noqa: F401

# Нормализация и календаризация 1m
from .core.validate import (  # noqa: F401
    ValidateStats,
    normalize_ohlcv_1m,
    align_and_flag_gaps,
    build_minute_calendar,
)

# Ресемплинг 1m → 5m/15m/1h
from .core.resample import resample_1m_to  # noqa: F401

# Parquet‑хранилище
from .io.parquet_store import write_idempotent, parquet_path  # noqa: F401

# Хвостовой кэш (sidecar latest.json)
from .io.tail_cache import TailInfo, read as tail_read, write as tail_write, update as tail_update, refresh_from_parquet  # noqa: F401

# DataReader для выборок по времени
from .datareader import DataReader, ReadStats  # noqa: F401

__all__ = [
    "__version__",
    "build_signature",
    # CLI
    "cli_main",
    # HTTP
    "BybitClient",
    "FetchStats",
    # Validate / Calendar
    "ValidateStats",
    "normalize_ohlcv_1m",
    "align_and_flag_gaps",
    "build_minute_calendar",
    # Resample
    "resample_1m_to",
    # Store
    "write_idempotent",
    "parquet_path",
    # Tail cache
    "TailInfo",
    "tail_read",
    "tail_write",
    "tail_update",
    "refresh_from_parquet",
    # Reader
    "DataReader",
    "ReadStats",
]
