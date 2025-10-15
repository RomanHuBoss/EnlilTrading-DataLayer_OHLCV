# Мэппинг таймфреймов и утилиты.
from datetime import timedelta

TF_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
}

def tf_to_pandas_rule(tf: str) -> str:
    if tf not in TF_MINUTES:
        raise ValueError(f"Неизвестный таймфрейм: {tf}")
    # Использовать новые алиасы pandas: min/h
    return {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "1h",
    }[tf]

def tf_minutes(tf: str) -> int:
    return TF_MINUTES[tf]
