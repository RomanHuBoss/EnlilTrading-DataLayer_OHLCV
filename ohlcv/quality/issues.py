# ohlcv/quality/issues.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
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
# Модель и реестр кодов C2
# =============================

@dataclass(frozen=True)
class IssueSpec:
    code: str
    desc: str
    severity: str      # "crit" | "high" | "medium" | "low"
    action: str        # "flag" | "fix" | "gapify" | "drop"
    level: int         # группировка для отчётов/приоритетов


# Канонический реестр по постановке C2
ISSUE_REGISTRY: Dict[str, IssueSpec] = {
    # Время/структура
    "R01_DUP_TS":        IssueSpec("R01_DUP_TS", "duplicate timestamps", "high", "drop", 1),
    "R02_TS_FUTURE":     IssueSpec("R02_TS_FUTURE", "timestamp in the future", "high", "drop", 1),
    "R03_TS_MISALIGNED": IssueSpec("R03_TS_MISALIGNED", "timestamp not aligned to TF", "medium", "fix", 1),

    # Инварианты OHLC/объём
    "R10_NEG_PRICE":     IssueSpec("R10_NEG_PRICE", "non-positive price in OHLC", "crit", "gapify", 2),
    "R11_NEG_VOL":       IssueSpec("R11_NEG_VOL", "negative volume", "medium", "fix", 2),
    "R12_NAN":           IssueSpec("R12_NAN", "NaN inside OHLC", "high", "gapify", 2),
    "R13_OHLC_ORDER":    IssueSpec("R13_OHLC_ORDER", "OHLC order violation", "high", "fix", 2),
    "R14_H_LT_L":        IssueSpec("R14_H_LT_L", "high < low", "high", "fix", 2),

    # Статистика
    "R20_VOL_SPIKE":     IssueSpec("R20_VOL_SPIKE", "abnormal volume spike vs baseline", "low", "flag", 3),
    "R21_ATR_SPIKE":     IssueSpec("R21_ATR_SPIKE", "abnormal ATR spike vs baseline", "low", "flag", 3),
    "R22_RET_SPIKE":     IssueSpec("R22_RET_SPIKE", "abnormal return spike vs baseline", "low", "flag", 3),
    "R23_ZERO_RUN":      IssueSpec("R23_ZERO_RUN", "long run of zero volume or flat bars", "low", "flag", 3),

    # Согласованность TF vs 1m
    "R30_OHLC_MISMATCH": IssueSpec("R30_OHLC_MISMATCH", "OHLC mismatch vs 1m aggregation", "high", "flag", 4),
    "R31_VOL_MISMATCH":  IssueSpec("R31_VOL_MISMATCH", "Volume mismatch vs 1m aggregation", "medium", "flag", 4),
    "R32_RANGE_MISMATCH":IssueSpec("R32_RANGE_MISMATCH", "Range outside minute range", "high", "flag", 4),
    "R33_COUNT_MISMATCH":IssueSpec("R33_COUNT_MISMATCH", "Insufficient minute bars in window", "medium", "flag", 4),
}

# Допустимые псевдонимы (регистр/подстроки)
CODE_ALIASES: Dict[str, str] = {
    # базовые
    "dup_ts": "R01_DUP_TS",
    "duplicate_ts": "R01_DUP_TS",
    "ts_future": "R02_TS_FUTURE",
    "future": "R02_TS_FUTURE",
    "misaligned": "R03_TS_MISALIGNED",
    "ts_misaligned": "R03_TS_MISALIGNED",
    # инварианты
    "neg_price": "R10_NEG_PRICE",
    "neg_vol": "R11_NEG_VOL",
    "nan": "R12_NAN",
    "ohlc_order": "R13_OHLC_ORDER",
    "h_lt_l": "R14_H_LT_L",
    # статистика
    "vol_spike": "R20_VOL_SPIKE",
    "atr_spike": "R21_ATR_SPIKE",
    "ret_spike": "R22_RET_SPIKE",
    "zero_run": "R23_ZERO_RUN",
    # согласованность
    "ohlc_mismatch": "R30_OHLC_MISMATCH",
    "vol_mismatch": "R31_VOL_MISMATCH",
    "range_mismatch": "R32_RANGE_MISMATCH",
    "count_mismatch": "R33_COUNT_MISMATCH",
}


def canonical_code(code: Any) -> str:
    """Канонизация кода. Неизвестные возвращаются как верхний регистр без пробелов."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "UNKNOWN"
    s = str(code).strip()
    if not s:
        return "UNKNOWN"
    s_up = s.upper()
    if s_up in ISSUE_REGISTRY:
        return s_up
    key = s.strip().lower().replace(" ", "").replace("-", "_")
    if key in CODE_ALIASES:
        return CODE_ALIASES[key]
    for k, v in CODE_ALIASES.items():
        if k in key:
            return v
    return s_up


# =============================
# Нормализация и сводки
# =============================

def normalize_issues_df(df: pd.DataFrame) -> pd.DataFrame:
    """Контракт C2. Выходные колонки: ts:int64, code:str, note:str, symbol:str, tf:str.
    Правила: канонизация code, типы, пустые строки вместо NaN, сортировка, дедуп (ts,code,note,symbol,tf).
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["ts", "code", "note", "symbol", "tf"]).astype(
            {"ts": "int64", "code": "string", "note": "string", "symbol": "string", "tf": "string"}
        )

    out = df.copy()

    # алиасы колонок
    ren = {}
    for c in out.columns:
        lc = str(c).strip().lower()
        if lc == "timestamp_ms":
            ren[c] = "ts"
        elif lc in ("issue", "err", "error", "code"):
            ren[c] = "code"
        elif lc in ("message", "msg", "note", "reason"):
            ren[c] = "note"
        elif lc in ("sym", "ticker", "symbol"):
            ren[c] = "symbol"
        elif lc in ("timeframe", "tf"):
            ren[c] = "tf"
    if ren:
        out = out.rename(columns=ren)

    # обязательные ts/code
    if "ts" not in out.columns or "code" not in out.columns:
        if "ts" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
            idx = out.index.tz_convert("UTC") if out.index.tz is not None else out.index.tz_localize("UTC")
            out["ts"] = (idx.asi8 // 1_000_000).astype("int64")
        if "code" not in out.columns and "issue" in out.columns:
            out = out.rename(columns={"issue": "code"})
    if "ts" not in out.columns or "code" not in out.columns:
        return pd.DataFrame(columns=["ts", "code", "note", "symbol", "tf"]).astype(
            {"ts": "int64", "code": "string", "note": "string", "symbol": "string", "tf": "string"}
        )

    # типы/канонизация
    out["ts"] = pd.to_numeric(out["ts"], errors="coerce").astype("Int64").fillna(0).astype("int64")
    out["code"] = out["code"].map(canonical_code).astype("string")

    # необязательные
    for col in ("note", "symbol", "tf"):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype("string").fillna("")

    # сортировка/дедуп
    out = out[["ts", "code", "note", "symbol", "tf"]]
    out = out.sort_values(["ts", "code"]).drop_duplicates(subset=["ts", "code", "note", "symbol", "tf"]).reset_index(drop=True)
    return out


def summarize_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Сводная таблица по кодам: count, first_ts, last_ts, severity, action, level."""
    nd = normalize_issues_df(df)
    if len(nd) == 0:
        return pd.DataFrame(columns=["code", "count", "first_ts", "last_ts", "severity", "action", "level"]).astype(
            {"code": "string", "count": "int64", "first_ts": "int64", "last_ts": "int64", "severity": "string", "action": "string", "level": "int64"}
        )
    g = nd.groupby("code")
    agg = g["ts"].agg(["count", "min", "max"]).rename(columns={"min": "first_ts", "max": "last_ts"}).reset_index()
    meta = pd.DataFrame.from_records(
        [vars(ISSUE_REGISTRY.get(c, IssueSpec(c, "unknown", "low", "flag", 9))) for c in agg["code"].astype(str)]
    )
    out = pd.merge(agg, meta[["code", "severity", "action", "level"]], on="code", how="left")
    out["first_ts"] = out["first_ts"].astype("int64")
    out["last_ts"] = out["last_ts"].astype("int64")
    out["level"] = out["level"].fillna(9).astype("int64")
    out["severity"] = out["severity"].fillna("low").astype("string")
    out["action"] = out["action"].fillna("flag").astype("string")
    out["code"] = out["code"].astype("string")
    return out.sort_values(["level", "code"]).reset_index(drop=True)


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
