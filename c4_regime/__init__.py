# c4_regime/__init__.py
from __future__ import annotations

"""Совместимость: перенаправление на ohlcv.regime.

Оставлено для старых импортов `from c4_regime import infer_regime`.
Новая точка: `from ohlcv.regime import RegimeConfig, infer_regime`.
"""
try:  # pragma: no cover
    from ohlcv.regime import RegimeConfig, infer_regime  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("ohlcv.regime недоступен: " + str(e))

__all__ = ["RegimeConfig", "infer_regime"]
