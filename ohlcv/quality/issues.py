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
# Спецификация issue
# =============================
@dataclass(frozen=True)
class IssueSpec:
    code: str
    description: str
    severity: str  # "low" | "medium" | "high" | "crit"
    action: str    # "flag" | "drop" | "fix" | "gapify" | "synthetic_flat_bar"
    dq_rank: int   # 1=высший приоритет


# =============================
# Реестр кодов C2
# =============================
ISSUE_REGISTRY: Dict[str, IssueSpec] = {
    # Время/структура
    "R01_DUP_TS": IssueSpec("R01_DUP_TS", "duplicate timestamp (last wins)", "high", "drop", 1),
    "R02_TS_FUTURE": IssueSpec("R02_TS_FUTURE", "bar in the future (beyond window)", "crit", "drop", 1),
    "R03_TS_MISALIGNED": IssueSpec("R03_TS_MISALIGNED", "timestamp misaligned to tf", "medium", "fix", 2),

    # Инварианты OHLC/объём
    "R10_NEG_PRICE": IssueSpec("R10_NEG_PRICE", "non-positive price in OHLC", "crit", "gapify", 1),
    "R11_NEG_VOL": IssueSpec("R11_NEG_VOL", "negative volume", "medium", "fix", 2),
    "R12_NAN": IssueSpec("R12_NAN", "NaN inside OHLC", "high", "gapify", 1),
    "R13_OHLC_ORDER": IssueSpec("R13_OHLC_ORDER", "OHLC order violation", "high", "fix", 2),
    "R14_H_LT_L": IssueSpec("R14_H_LT_L", "high < low", "high", "fix", 2),

    # Статистика
    "R20_VOL_SPIKE": IssueSpec("R20_VOL_SPIKE", "abnormal volume spike vs baseline", "low", "flag", 3),
    "R21_ATR_SPIKE": IssueSpec("R21_ATR_SPIKE", "abnormal ATR spike vs baseline", "low", "flag", 3),
    "R22_RET_SPIKE": IssueSpec("R22_RET_SPIKE", "abnormal return spike vs baseline", "low", "flag", 3),
    "R23_ZERO_RUN": IssueSpec("R23_ZERO_RUN", "long run of zero volume", "low", "flag", 3),

    # Согласованность TF против 1m
    "R30_OHLC_MISMATCH": IssueSpec("R30_OHLC_MISMATCH", "aggregated ohlc mismatch vs 1m", "medium", "flag", 3),
    "R31_VOL_MISMATCH": IssueSpec("R31_VOL_MISMATCH", "aggregated volume mismatch vs 1m", "medium", "flag", 3),
    "R32_RANGE_MISMATCH": IssueSpec("R32_RANGE_MISMATCH", "high/low outside minute range", "high", "flag", 2),
    "R33_COUNT_MISMATCH": IssueSpec("R33_COUNT_MISMATCH", "insufficient minute count in window", "medium", "flag", 3),

    # Служебные
    "MISSING_FILLED": IssueSpec("MISSING_FILLED", "gap filled by synthesizer (flat bar)", "low", "synthetic_flat_bar", 2),
}


# Алиасы для совместимости с разными источниками
CODE_ALIASES: Dict[str, str] = {
    # R01–R03
    "DUP_TS": "R01_DUP_TS",
    "TS_FUTURE": "R02_TS_FUTURE",
    "TS_MISALIGNED": "R03_TS_MISALIGNED",
    # R10–R14
    "NEG_PRICE": "R10_NEG_PRICE",
    "NEG_VOL": "R11_NEG_VOL",
    "NAN_OHLC": "R12_NAN",
    "OHLC_ORDER": "R13_OHLC_ORDER",
    "H_LT_L": "R14_H_LT_L",
    # R20–R23
    "VOL_SPIKE": "R20_VOL_SPIKE",
    "ATR_SPIKE": "R21_ATR_SPIKE",
    "RET_SPIKE": "R22_RET_SPIKE",
    "ZERO_RUN": "R23_ZERO_RUN",
    # R30–R33
    "OHLC_MISMATCH": "R30_OHLC_MISMATCH",
    "VOL_MISMATCH": "R31_VOL_MISMATCH",
    "RANGE_MISMATCH": "R32_RANGE_MISMATCH",
    "COUNT_MISMATCH": "R33_COUNT_MISMATCH",
    # Служебные
    "GAP_FILLED": "MISSING_FILLED",
}


# =============================
# Утилиты нормализации/сводки
# =============================
_DEF_COLS = ["ts", "code", "note", "severity", "action", "dq_rank", "symbol", "tf"]


def canonical_code(code: str) -> str:
    c = (code or "").strip().upper()
    if c in ISSUE_REGISTRY:
        return c
    if c in CODE_ALIASES:
        return CODE_ALIASES[c]
    return c  # неизвестный — оставляем как есть


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in _DEF_COLS:
        if c not in out.columns:
            out[c] = None
    return out[_DEF_COLS]


def normalize_issues_df(df: pd.DataFrame) -> pd.DataFrame:
    """Приведение issues к канону: типы, алиасы, заполнение spec, порядок и сортировка."""
    if len(df) == 0:
        return _ensure_columns(pd.DataFrame(columns=_DEF_COLS))
    out = _ensure_columns(df)

    # Типы
    out["ts"] = pd.to_numeric(out["ts"], errors="coerce").astype("Int64").astype("float").astype("int64")
    out["code"] = out["code"].astype(str).str.upper().map(canonical_code)
    out["note"] = out["note"].astype("string").fillna(pd.NA)
    out["symbol"] = out["symbol"].astype("string").fillna(pd.NA)
    out["tf"] = out["tf"].astype("string").fillna(pd.NA)

    # Заполнение по реестру
    def _fill_from_spec(row: pd.Series) -> pd.Series:
        spec = ISSUE_REGISTRY.get(row["code"])  # type: ignore[index]
        if spec is None:
            # дефолты для неизвестных кодов
            row["severity"] = row["severity"] if pd.notna(row["severity"]) else "low"
            row["action"] = row["action"] if pd.notna(row["action"]) else "flag"
            row["dq_rank"] = int(row["dq_rank"]) if pd.notna(row["dq_rank"]) else 5
            return row
        row["severity"] = row["severity"] if pd.notna(row["severity"]) else spec.severity
        row["action"] = row["action"] if pd.notna(row["action"]) else spec.action
        row["dq_rank"] = int(row["dq_rank"]) if pd.notna(row["dq_rank"]) else spec.dq_rank
        return row

    out = out.apply(_fill_from_spec, axis=1)

    # Удаление точных дублей и сортировка
    out = out.drop_duplicates(subset=["ts", "code", "note", "symbol", "tf"]).sort_values(["ts", "dq_rank"]).reset_index(drop=True)

    return out[_DEF_COLS]


def summarize_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Сводка по кодам и уровню: counts, доля, ранжирование."""
    norm = normalize_issues_df(df)
    if len(norm) == 0:
        return pd.DataFrame(columns=["code", "count", "dq_rank"]).astype({"count": "int64", "dq_rank": "int64"})
    grp = norm.groupby(["code", "dq_rank"], as_index=False).size().rename(columns={"size": "count"})
    return grp.sort_values(["dq_rank", "count", "code"], ascending=[True, False, True]).reset_index(drop=True)


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
