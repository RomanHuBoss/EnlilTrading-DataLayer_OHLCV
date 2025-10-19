from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "META_COLS",
    "FeatureSchema",
    "infer_feature_cols",
    "reorder_columns",
    "validate_features_df",
]

# =============================
# Канон C3
# =============================
META_COLS: Tuple[str, ...] = (
    "ts",              # int64
    "symbol",          # string
    "tf",              # string
    "f_valid_from",    # int64
    "f_build_version", # string
)


@dataclass(frozen=True)
class FeatureSchema:
    meta: Tuple[str, ...] = META_COLS
    # Остальные — любые столбцы с префиксом f_

    def all(self, fcols: Iterable[str]) -> List[str]:
        return list(self.meta) + [c for c in fcols if c not in self.meta]


# =============================
# Вспомогательные
# =============================

def infer_feature_cols(df: pd.DataFrame) -> List[str]:
    """Возвращает список фич, начинающихся с f_. Стабильный (отсортированный) порядок."""
    fcols = sorted([c for c in df.columns if c.startswith("f_")])
    return fcols


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Выставляет канонический порядок: META_COLS → сортированные f_* → прочие (если есть)."""
    cols_meta = [c for c in META_COLS if c in df.columns]
    fcols = infer_feature_cols(df)
    others = [c for c in df.columns if c not in set(cols_meta) | set(fcols)]
    order = cols_meta + fcols + others
    return df[order]


# =============================
# Валидатор/нормализатор
# =============================

def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # ts / f_valid_from → int64
    for c in ["ts", "f_valid_from"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64").astype("float").astype("int64")
    # string-поля
    for c in ["symbol", "tf", "f_build_version"]:
        if c in out.columns:
            out[c] = out[c].astype("string").fillna(pd.NA)
    # фичи → float64
    for c in infer_feature_cols(out):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out


def validate_features_df(df: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Приводит DataFrame с фичами к канону C3.

    Поведение:
    - Проверяет наличие META_COLS; если strict=True — все обязательные поля должны присутствовать.
    - Приводит типы: ts/f_valid_from → int64; все f_* → float64; строки → pandas-string.
    - Сортирует столбцы в каноническом порядке.
    - Сортирует строки по ts, снимает дубликаты ts (последний выигрывает).
    """
    missing = [c for c in META_COLS if c not in df.columns]
    if strict and missing:
        raise ValueError(f"отсутствуют обязательные столбцы: {missing}")

    out = _coerce_types(df)

    # Стабилизация строк
    if "ts" in out.columns:
        out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)

    # Порядок колонок
    out = reorder_columns(out)
    return out
