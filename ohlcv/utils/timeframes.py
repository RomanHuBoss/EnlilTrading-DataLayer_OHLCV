# Мэппинг таймфреймов и утилиты.
from datetime import timedelta

TF_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
}

def tf_to_pandas_rule(tf: str) -> str:
    """Возврат правила ресемплинга pandas для таймфрейма."""
    if tf not in TF_MINUTES:
        raise ValueError(f"Неизвестный таймфрейм: {tf}")
    # Используем минутные/часовые границы, закрываем справа [right-closed].
    rule = {"1m": "1T", "5m": "5T", "15m": "15T", "1h": "1H"}[tf]
    return rule

def tf_minutes(tf: str) -> int:
    return TF_MINUTES[tf]
