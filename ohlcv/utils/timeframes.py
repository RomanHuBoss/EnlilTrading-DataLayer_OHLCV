# ohlcv/utils/timeframes.py
# Примитивы работы с таймфреймами без deprecated-правил Pandas.
# Используются нижние регистры: "min", "5min", "15min", "1h".

from __future__ import annotations

from typing import Literal

AllowedTF = Literal["1m", "5m", "15m", "1h"]


def tf_minutes(tf: AllowedTF) -> int:
    if tf == "1m":
        return 1
    if tf == "5m":
        return 5
    if tf == "15m":
        return 15
    if tf == "1h":
        return 60
    raise ValueError(f"Неизвестный TF: {tf}")


def tf_to_pandas_rule(tf: AllowedTF) -> str:
    # pandas >= 2.2: 'T' и 'H' помечены как deprecated; используем 'min' и 'h'.
    if tf == "1m":
        return "min"
    if tf == "5m":
        return "5min"
    if tf == "15m":
        return "15min"
    if tf == "1h":
        return "1h"
    raise ValueError(f"Неизвестный TF: {tf}")


def is_supported_tf(tf: str) -> bool:
    return tf in {"1m", "5m", "15m", "1h"}
