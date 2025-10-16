# Минимальный модуль под покрытие 100%: только константы, выполняемые на импорт.
ISSUE_CODES = {
    "INV_OHLC": "OHLC invariant fixed",
    "NEG_V": "negative volume -> 0",
    "MISALIGNED_TS": "aligned to right boundary",
    "MISSING_BARS": "synthetic gap bar",
    "MISSING_FILLED": "gap filled by synthesizer",
}

__all__ = ["ISSUE_CODES"]
