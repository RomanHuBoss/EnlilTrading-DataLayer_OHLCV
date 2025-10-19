from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Any

import pandas as pd

__all__ = [
    "IssueSpec",
    "ISSUE_REGISTRY",
    "CODE_ALIASES",
    "canonical_code",
    "normalize_issues_df",
    "summarize_issues",
    "merge_issues",
]


# =============================
# Спецификация и реестр кодов C2
# =============================

@dataclass(frozen=True)
class IssueSpec:
    code: str
    description: str
    severity: str  # low|medium|high
    action: str    # описание корректирующего действия/рекомендации
    dq_rank: int   # приоритет сортировки (чем больше, тем важнее)


# Канонический реестр. Значения dq_rank согласованы с validator.py
ISSUE_REGISTRY: Dict[str, IssueSpec] = {
    # Базовые
    "NEG_V": IssueSpec(
        code="NEG_V",
        description="negative volume observed; clipped to zero",
        severity="medium",
        action="clip_to_zero",
        dq_rank=30,
    ),
    "INV_OHLC": IssueSpec(
        code="INV_OHLC",
        description="OHLC invariant violated; bounds fixed",
        severity="high",
        action="fix_bounds",
        dq_rank=40,
    ),
    "MISALIGNED_TS": IssueSpec(
        code="MISALIGNED_TS",
        description="timestamp misaligned; aligned to right boundary",
        severity="low",
        action="align_right",
        dq_rank=5,
    ),
    "MISSING": IssueSpec(
        code="MISSING",
        description="calendar gap; synthetic minute inserted",
        severity="low",
        action="synthetic_flat_bar",
        dq_rank=1,
    ),
    "MISSING_FILLED": IssueSpec(
        code="MISSING_FILLED",
        description="gap filled by synthesizer (flat bar)",
        severity="low",
        action="synthetic_flat_bar",
        dq_rank=2,
    ),

    # R20–R22 статистические флаги
    "R20_VOL_SPIKE": IssueSpec(
        code="R20_VOL_SPIKE",
        description="abnormal volume spike vs rolling baseline",
        severity="low",
        action="flag",
        dq_rank=10,
    ),
    "R21_ATR_SPIKE": IssueSpec(
        code="R21_ATR_SPIKE",
        description="abnormal ATR spike vs rolling baseline",
        severity="medium",
        action="flag",
        dq_rank=20,
    ),
    "R22_RET_SPIKE": IssueSpec(
        code="R22_RET_SPIKE",
        description="abnormal absolute return spike",
        severity="medium",
        action="flag",
        dq_rank=20,
    ),

    # R30–R33 согласованность производных ТФ с 1m
    "R30_OHLC_MISMATCH": IssueSpec(
        code="R30_OHLC_MISMATCH",
        description="aggregated OHLC differs from resampled 1m",
        severity="high",
        action="rebuild_from_1m",
        dq_rank=90,
    ),
    "R31_VOL_MISMATCH": IssueSpec(
        code="R31_VOL_MISMATCH",
        description="aggregated volume differs from 1m sum",
        severity="medium",
        action="rebuild_from_1m",
        dq_rank=70,
    ),
    "R33_COUNT_MISMATCH": IssueSpec(
        code="R33_COUNT_MISMATCH",
        description="missing aggregated bar vs 1m calendar",
        severity="high",
        action="reindex_and_fill",
        dq_rank=80,
    ),
}


# Синонимы/наследие → канон
CODE_ALIASES: Dict[str, str] = {
    # Базовые
    "inv_ohlc": "INV_OHLC",
    "gap": "MISSING",
    "missing": "MISSING",
    "neg_v": "NEG_V",
    "misaligned_ts": "MISALIGNED_TS",
    "missing_filled": "MISSING_FILLED",
    # Короткие
    "VOL_SPIKE": "R20_VOL_SPIKE",
    "ATR_SPIKE": "R21_ATR_SPIKE",
    "RET_SPIKE": "R22_RET_SPIKE",
}


def canonical_code(code: Optional[str]) -> Optional[str]:
    if code is None:
        return None
    key = str(code).strip()
    if key in ISSUE_REGISTRY:
        return key
    key_up = key.upper()
    if key_up in ISSUE_REGISTRY:
        return key_up
    return CODE_ALIASES.get(key, CODE_ALIASES.get(key_up, key_up))


# =============================
# Нормализация и сводки
# =============================

_DEF_COLS = ["ts", "code", "note", "severity", "action", "dq_rank", "symbol", "tf"]


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in _DEF_COLS:
        if c not in out.columns:
            out[c] = None
    return out[_DEF_COLS]


def normalize_issues_df(df: pd.DataFrame) -> pd.DataFrame:
    """Приведение issues к каноническому виду с заполнением значений из реестра.

    - ts → datetime64[ns, UTC]
    - code → канон + в верхнем регистре
    - заполнение severity/action/dq_rank из ISSUE_REGISTRY, если не указаны
    - сортировка по ts, dq_rank desc, code
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=_DEF_COLS)

    out = _ensure_columns(df)

    # ts → UTC
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")

    # code → канон
    out["code"] = out["code"].map(canonical_code)

    # Подстановка значений из реестра
    def _fill_row(row: pd.Series) -> pd.Series:
        spec = ISSUE_REGISTRY.get(row["code"])  # уже upper/canonical
        if spec is None:
            # неизвестный код — мягкая деградация
            row["severity"] = row.get("severity") or "low"
            row["action"] = row.get("action") or "flag"
            row["dq_rank"] = row.get("dq_rank") or 0
            return row
        if not row.get("severity"):
            row["severity"] = spec.severity
        if not row.get("action"):
            row["action"] = spec.action
        if not row.get("dq_rank"):
            row["dq_rank"] = spec.dq_rank
        return row

    out = out.apply(_fill_row, axis=1)

    # Категории для компактности и консистентности
    out["severity"] = pd.Categorical(out["severity"], categories=["low", "medium", "high"], ordered=True)

    # Сортировка: время↑, важность↓, код↑
    out = out.sort_values(["ts", "dq_rank", "code"])  # dq_rank чем больше — тем важнее; итог читается слева-направо
    out = out.reset_index(drop=True)
    return out


def summarize_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Групповая сводка по code/severity/action с подсчётом штук."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["code", "severity", "action", "count"]).astype({"count": "int64"})
    g = (
        df.groupby(["code", "severity", "action"], dropna=False)
          .size()
          .rename("count")
          .reset_index()
          .sort_values(["count", "code"], ascending=[False, True])
    )
    g["count"] = g["count"].astype("int64")
    return g


def merge_issues(base: pd.DataFrame, add: pd.DataFrame) -> pd.DataFrame:
    """Слияние двух наборов issues с нормализацией и удалением точных дублей."""
    b = normalize_issues_df(base)
    a = normalize_issues_df(add)
    if len(b) == 0:
        return a
    if len(a) == 0:
        return b
    out = pd.concat([b, a], ignore_index=True)
    out = out.drop_duplicates(subset=["ts", "code", "note", "symbol", "tf"]).reset_index(drop=True)
    out = normalize_issues_df(out)
    return out
