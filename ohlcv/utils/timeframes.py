# ohlcv/utils/timeframes.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

# Базовые константы
SECOND_MS = 1_000
MINUTE_MS = 60 * SECOND_MS
HOUR_MS = 60 * MINUTE_MS

# Поддерживаемые таймфреймы → миллисекунды
_TF_MS = {
    "1m": MINUTE_MS,
    "1min": MINUTE_MS,
    "60s": MINUTE_MS,
    "5m": 5 * MINUTE_MS,
    "15m": 15 * MINUTE_MS,
    "1h": HOUR_MS,
    "60m": HOUR_MS,
}

# Соответствие pandas offset-алиасов
_TF_FREQ = {
    "1m": "1T",
    "1min": "1T",
    "60s": "1T",
    "5m": "5T",
    "15m": "15T",
    "1h": "1H",
    "60m": "1H",
}


def parse_tf(tf: str) -> int:
    """Возвращает размер таймфрейма в миллисекундах. Ошибка при неподдерживаемом значении."""
    key = tf.strip().lower()
    if key not in _TF_MS:
        raise ValueError(f"неподдерживаемый таймфрейм: {tf}")
    return _TF_MS[key]


def to_pandas_freq(tf: str) -> str:
    key = tf.strip().lower()
    if key not in _TF_FREQ:
        raise ValueError(f"неподдерживаемый таймфрейм: {tf}")
    return _TF_FREQ[key]


def is_multiple_of_1m(tf: str) -> bool:
    return parse_tf(tf) % MINUTE_MS == 0


def floor_ts_ms(ts_ms: int, tf_ms: int) -> int:
    """Округляет timestamp вниз к левому краю окна tf_ms (эпоха UTC как якорь)."""
    if tf_ms <= 0:
        raise ValueError("tf_ms must be > 0")
    return (ts_ms // tf_ms) * tf_ms


def ceil_ts_ms(ts_ms: int, tf_ms: int) -> int:
    """Округляет timestamp вверх к левому краю следующего окна tf_ms (если уже кратен — вернёт тот же)."""
    if tf_ms <= 0:
        raise ValueError("tf_ms must be > 0")
    return ((ts_ms + tf_ms - 1) // tf_ms) * tf_ms


def parent_start_1m_to(ts_ms: int, dst_tf: str) -> int:
    """Левый край окна dst_tf для минутного бара с началом в ts_ms."""
    tf_ms = parse_tf(dst_tf)
    if tf_ms % MINUTE_MS != 0:
        raise ValueError("целевой tf должен быть кратен 1 минуте")
    return floor_ts_ms(ts_ms, tf_ms)


def align_range_ms(start_ms: int, end_ms: int, tf_ms: int) -> Tuple[int, int]:
    """Выравнивает диапазон [start, end) к сетке tf_ms.

    Левая граница округляется вниз, правая — вверх, результат сохраняет полуинтервал.
    При end<=start возвращает (start, start).
    """
    if end_ms <= start_ms:
        return start_ms, start_ms
    s = floor_ts_ms(start_ms, tf_ms)
    e = ceil_ts_ms(end_ms, tf_ms)
    return s, e


def day_bounds_utc(ts_ms: int) -> Tuple[int, int]:
    """Границы календарного дня UTC для метки ts_ms. Возвращает [start_ms, end_ms)."""
    ts = pd.to_datetime(int(ts_ms), unit="ms", utc=True)
    start = ts.floor("D")
    end = start + pd.Timedelta(days=1)
    return int(start.value // 1_000_000), int(end.value // 1_000_000)


def ensure_minute_aligned_index(idx: pd.DatetimeIndex) -> None:
    """Проверка, что индекс выровнен по минутам (секунды=0, нс=0) и tz-aware UTC."""
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
        raise TypeError("ожидается tz-aware DatetimeIndex (UTC)")
    if not (idx.tz == pd.Timestamp(0, tz="UTC").tz):
        # Приведение к UTC должно делаться выше по стеку — здесь только валидация
        raise TypeError("DatetimeIndex должен быть в UTC")
    if not (idx.second == 0).all() or not (idx.nanosecond == 0).all():
        raise ValueError("индекс не выровнен по минутным границам")


def minute_index(start_ms: int, end_ms: int) -> pd.DatetimeIndex:
    """Минутная шкала [start, end) UTC. Удобно для построения календаря."""
    if end_ms <= start_ms:
        return pd.DatetimeIndex([], name="ts", tz="UTC")
    s = floor_ts_ms(start_ms, MINUTE_MS)
    e = ceil_ts_ms(end_ms, MINUTE_MS)
    rng = pd.date_range(pd.to_datetime(s, unit="ms", utc=True), pd.to_datetime(e, unit="ms", utc=True), freq="1T", inclusive="left")
    return pd.DatetimeIndex(rng, name="ts")
