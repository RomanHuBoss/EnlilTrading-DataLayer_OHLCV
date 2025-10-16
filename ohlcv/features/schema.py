from __future__ import annotations

import pandas as pd
from typing import Dict

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

# Допустимая внутренняя альтернативная схема (C1/C2)
ALT_MAP = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "turnover"}

REQUIRED = set(EXPECTED_DTYPES.keys())


class SchemaError(ValueError):
    pass


def normalize_and_validate(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """
    - Переименует альтернативные колонки в канонические.
    - Приведёт dtypes к EXPECTED_DTYPES (опциональная 'turnover' — best-effort).
    - При strict=True: проверит наличие всех REQUIRED и отсутствие бесконечностей/NaN в базовых колонках.
    """
    # Автопереименование схемы C1/C2 → канон
    cols = set(df.columns)
    if {"o", "h", "l", "c", "v"}.issubset(cols):
        df = df.rename(columns={k: v for k, v in ALT_MAP.items() if k in df.columns})

    # Наличие обязательных колонок
    miss = REQUIRED - set(df.columns)
    if miss:
        raise SchemaError(f"Отсутствуют обязательные колонки: {sorted(miss)}")

    # Приведение типов
    for c, dt in EXPECTED_DTYPES.items():
        if dt == "int64":
            df[c] = (
                pd.to_numeric(df[c], errors="coerce")
                .astype("Int64")
                .astype("int64", errors="ignore")
            )
        elif dt.startswith("float"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = df[c].astype("string")

    if "turnover" in df.columns:
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")

    if strict:
        base = ["open", "high", "low", "close", "volume"]
        bad = df[base].isna().any(axis=1)
        if bad.any():
            n = int(bad.sum())
            raise SchemaError(f"Найдены {n} строк(и) с NaN в базовых колонках {base}")

    return df
