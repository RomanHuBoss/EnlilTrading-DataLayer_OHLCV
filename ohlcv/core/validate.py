# Валидация рядов и контроль пропусков.
import pandas as pd

def validate_1m_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Индекс должен быть DatetimeIndex")
    if df.index.tz is None or str(df.index.tz) != "UTC":
        raise ValueError("Индекс должен быть в UTC (tz-aware)")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Индекс должен быть монотонно возрастающим")
    if df.index.duplicated().any():
        raise ValueError("Обнаружены дубликаты таймстампов")
    # Кратность минуте
    if ((df.index.view("i8") // 10**9) % 60).any():
        raise ValueError("Таймстампы не выровнены по минутам")

def missing_rate(df: pd.DataFrame, freq: str = "min") -> float:
    full = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
    rate = 1.0 - (len(df.index.unique()) / len(full))
    return max(0.0, rate)

def ensure_missing_threshold(df: pd.DataFrame, threshold: float = 0.0001) -> None:
    rate = missing_rate(df)
    if rate > threshold:
        raise ValueError(f"Доля пропусков {rate:.6f} превышает порог {threshold:.6f}")
