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
# Порядок по контракту: базовые колонки входа → symbol, tf → f_* → f_valid_from, f_build_version
META_COLS: Tuple[str, ...] = (
    "ts",              # int64 (UTC ms, начало бара)
    "symbol",          # string
    "tf",              # string
    "f_valid_from",    # int64 (индекс первого полного окна, где нет NaN во всех заявленных f_*)
    "f_build_version", # string
)

_RESERVED_F_META: Tuple[str, ...] = ("f_valid_from", "f_build_version")


@dataclass(frozen=True)
class FeatureSchema:
    meta: Tuple[str, ...] = META_COLS
    # Остальные — любые столбцы с префиксом f_

    def all(self, fcols: Iterable[str]) -> List[str]:
        return list(self.meta) + [c for c in fcols if c not in self.meta]


# =============================
# Вспомогательные
# =============================

_BASE_ORDER = [
    # Таймштамп и производные
    "ts", "timestamp_ms", "start_time_iso",
    # OHLCV по канону C1/C2
    "o", "h", "l", "c", "v", "t",
    # Альтернативные имена (на случай внешнего ввода)
    "open", "high", "low", "close", "volume", "turnover",
]


def infer_feature_cols(df: pd.DataFrame) -> List[str]:
    """Стабильный список фич с префиксом f_, отсортированный лексикографически.
    Исключаются служебные f_valid_from и f_build_version.
    """
    cols = []
    for c in df.columns:
        if not isinstance(c, str):
            continue
        if not c.startswith("f_"):
            continue
        if c in _RESERVED_F_META:
            continue
        cols.append(c)
    return sorted(cols)


def _order_base_cols(df: pd.DataFrame) -> List[str]:
    """Стабильный порядок для базовых колонок входа (оставляем только существующие)."""
    seen = []
    present = set(df.columns)
    for c in _BASE_ORDER:
        if c in present and c not in seen:
            seen.append(c)
    return seen


def _uniq_keep_order(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Выставляет канонический порядок C3:
    base(в порядке _BASE_ORDER, как есть) → symbol, tf → f_* (sorted) → f_valid_from, f_build_version → прочие.
    """
    cols = list(df.columns)
    base = _order_base_cols(df)

    st = [c for c in ("symbol", "tf") if c in cols]

    fcols = infer_feature_cols(df)

    tail_meta = [c for c in ("f_valid_from", "f_build_version") if c in cols]

    others = [c for c in cols if c not in set(base) | set(st) | set(fcols) | set(tail_meta)]

    order = _uniq_keep_order(base + st + fcols + tail_meta + others)
    return df[order]


# =============================
# Валидатор/нормализатор
# =============================

def _coerce_types(df: pd.DataFrame, *, strict: bool) -> pd.DataFrame:
    out = df.copy()
    # ts/f_valid_from → числовые
    if "ts" in out.columns:
        out["ts"] = pd.to_numeric(out["ts"], errors="coerce")
    if "f_valid_from" in out.columns:
        out["f_valid_from"] = pd.to_numeric(out["f_valid_from"], errors="coerce")

    if strict:
        if "ts" not in out.columns:
            raise ValueError("отсутствует обязательная колонка: ts")
        if out["ts"].isna().any():
            raise ValueError("ts содержит NaN после приведения типов")
        if "f_valid_from" in out.columns and out["f_valid_from"].isna().any():
            raise ValueError("f_valid_from содержит NaN после приведения типов")
    else:
        if "ts" in out.columns:
            out = out.loc[~out["ts"].isna()].copy()
        if "f_valid_from" in out.columns:
            out["f_valid_from"] = out["f_valid_from"].fillna(out.get("ts"))

    if "ts" in out.columns:
        out["ts"] = out["ts"].astype("int64")
    if "f_valid_from" in out.columns:
        out["f_valid_from"] = out["f_valid_from"].astype("int64")

    for c in ["symbol", "tf", "f_build_version"]:
        if c in out.columns:
            out[c] = out[c].astype("string")

    for c in infer_feature_cols(out):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")

    return out


def validate_features_df(df: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Приводит DataFrame с фичами к канону C3.

    Правила:
    - Наличие META_COLS при strict=True.
    - Типы: ts/f_valid_from → int64; все f_* → float64; строки → pandas-string.
    - Удаление дубликатов по ts (последний wins), сортировка по ts.
    - Порядок столбцов: base → symbol,tf → f_* → f_valid_from,f_build_version → прочие.
    """
    if strict:
        missing = [c for c in META_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"отсутствуют обязательные столбцы: {missing}")

    out = _coerce_types(df, strict=strict)

    if "ts" in out.columns:
        out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)

    out = reorder_columns(out)
    return out
