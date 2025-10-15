# Валидация и репарация OHLCV-рядов.
import pandas as pd
from typing import Tuple

def validate_1m_index(df: pd.DataFrame) -> None:
    """Требования к 1m ряду: tz-aware UTC, монотонность, без дублей, выравнивание по минутам."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Индекс должен быть DatetimeIndex")
    if df.index.tz is None or str(df.index.tz) != "UTC":
        raise ValueError("Индекс должен быть в UTC (tz-aware)")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Индекс должен быть монотонно возрастающим")
    if df.index.duplicated().any():
        raise ValueError("Обнаружены дубликаты таймстампов")
    # кратность минуте
    if ((df.index.view("i8") // 10**9) % 60).any():
        raise ValueError("Таймстампы не выровнены по минутам")

def missing_rate(df: pd.DataFrame, freq: str = "min") -> float:
    """Доля пропусков относительно полной минутной сетки внутри диапазона df."""
    if df.empty:
        return 0.0
    full = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
    rate = 1.0 - (len(df.index.unique()) / len(full))
    return max(0.0, rate)

def ensure_missing_threshold(df: pd.DataFrame, threshold: float = 0.0001) -> None:
    """Исключение при превышении порога пропусков."""
    rate = missing_rate(df)
    if rate > threshold:
        raise ValueError(f"Доля пропусков {rate:.6f} превышает порог {threshold:.6f}")

def fill_1m_gaps(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Заполнение дыр в минутной сетке синтетическими барами:
      close: ffill предыдущего close (при необходимости bfill стартовой точки)
      open:  равен заполненному close
      high/low: равны open/close (плоская свеча)
      volume: 0.0
      turnover 't' (если есть): 0.0
    Возврат: (df_filled, n_filled). Индекс — tz-aware UTC минутные границы.
    """
    if df.empty:
        return df, 0
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError("Ожидается tz-aware DatetimeIndex (UTC)")

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="min", tz="UTC")
    out = df.reindex(full_idx)

    missing = out["c"].isna()
    if missing.any():
        base_close = out["c"].ffill().bfill()
        out.loc[missing, "c"] = base_close[missing]
        out.loc[missing, "o"] = base_close[missing]
        out.loc[missing, "h"] = base_close[missing]
        out.loc[missing, "l"] = base_close[missing]
        out.loc[missing, "v"] = 0.0
        if "t" in out.columns:
            out.loc[missing, "t"] = 0.0

    out = out.sort_index()
    return out, int(missing.sum())
