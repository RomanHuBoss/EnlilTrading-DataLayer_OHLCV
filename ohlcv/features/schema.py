from __future__ import annotations

from typing import Dict

import pandas as pd

# Каноническая схема входа для C3
EXPECTED_DTYPES: Dict[str, str] = {
    "timestamp_ms": "int64",
    "start_time_iso": "string",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
    # "turnover": "float64",  # опционально
}

REQUIRED_COLS = ["timestamp_ms", "start_time_iso", "open", "high", "low", "close", "volume"]

ALT_NAMES = {
    # иногда приходят короткие имена — нормализуем
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
}


class SchemaError(ValueError):
    pass


def normalize_schema(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Нормализация входной таблицы к канонической схеме.

    Действия:
    - Переименует альтернативные колонки в канонические.
    - Приведёт dtypes к EXPECTED_DTYPES (опциональная 'turnover' — best-effort).
    - При strict=True: проверит наличие всех REQUIRED и отсутствие NaN
      в базовых колонках.
    """
    # Переименование альтернативных колонок в канон
    rename_map = {c: ALT_NAMES[c] for c in df.columns if c in ALT_NAMES}
    if rename_map:
        df = df.rename(columns=rename_map)

    # Проверка обязательных колонок (в строгом режиме — падаем)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing and strict:
        raise SchemaError(f"Отсутствуют обязательные колонки: {missing}")

    # Приведение типов к ожидаемым
    for c, dt in EXPECTED_DTYPES.items():
        if c not in df.columns:
            continue
        if c == "timestamp_ms":
            df[c] = pd.to_numeric(df[c], errors="raise", downcast=None).astype("int64")
        elif dt == "float64":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        elif dt == "string":
            # string dtype (Arrow-friendly)
            df[c] = df[c].astype("string")
        else:
            df[c] = df[c].astype(dt)

    # Опциональная метрика оборота — мягкое приведение
    if "turnover" in df.columns:
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")

    if strict:
        base = ["open", "high", "low", "close", "volume"]
        bad = df[base].isna().any(axis=1)
        if bad.any():
            n = int(bad.sum())
            raise SchemaError(f"Найдены {n} строк(и) с NaN в базовых колонках {base}")

    return df
