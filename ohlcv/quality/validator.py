from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .issues import normalize_issues_df

__all__ = [
    "QualityConfig",
    "validate",
]

# =============================
# Конфигурация
# =============================
@dataclass
class QualityConfig:
    # Базовые допуски
    misaligned_tolerance_seconds: int = 1
    epsilon_rel: float = 1e-6  # допуск при сравнении чисел
    # Окна/пороги статистических правил
    vol_window: int = 50
    atr_window: int = 14
    ret_window: int = 50
    z_thr_vol: float = 6.0
    z_thr_atr: float = 6.0
    z_thr_ret: float = 8.0

# =============================
# Вспомогательные
# =============================
_DEF_COLS = ["o", "h", "l", "c", "v", "t", "is_gap"]


def _ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ts" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out["ts"] = (out.index.view("int64") // 1_000_000).astype("int64")
        else:
            raise ValueError("ожидался столбец 'ts' или DatetimeIndex")
    out["ts"] = pd.to_numeric(out["ts"], errors="coerce").astype("int64")
    for c in ["o", "h", "l", "c", "v", "t"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    if "is_gap" in out.columns:
        out["is_gap"] = out["is_gap"].astype(bool)
    # сортировка + дедуп (последний выигрывает)
    out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    return out


def _tf_ms(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1]) * 60_000
    if tf.endswith("h"):
        return int(tf[:-1]) * 60 * 60_000
    raise ValueError(f"неподдерживаемый tf: {tf}")


def _to_pandas_freq(tf: str) -> str:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return f"{int(tf[:-1])}T"
    if tf.endswith("h"):
        return f"{int(tf[:-1])}H"
    raise ValueError(f"неподдерживаемый tf: {tf}")


def _prev_close_series(df: pd.DataFrame) -> pd.Series:
    c = df["c"] if "c" in df.columns else pd.Series(np.nan, index=df.index)
    return c.shift(1).fillna(c)


def _append_issue(issues: List[Dict[str, Any]], ts: int, code: str, note: str | None = None) -> None:
    issues.append({"ts": int(ts), "code": code, "note": note})


def _rel_ne(a: pd.Series, b: pd.Series, eps: float) -> pd.Series:
    # относительное сравнение с допуском
    den = np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    return (np.abs(a - b) / den) > eps

# =============================
# Правила C2 (базовые)
# =============================

def _rule_R01_DUP_TS(df: pd.DataFrame, issues: List[Dict[str, Any]]) -> pd.DataFrame:
    dup = df["ts"].duplicated(keep="last")
    if dup.any():
        for ts in df.loc[dup, "ts"].tolist():
            _append_issue(issues, ts, "R01_DUP_TS")
        df = df.loc[~dup].copy()
    return df


def _rule_R02_TS_FUTURE(df: pd.DataFrame, tf_ms: int, issues: List[Dict[str, Any]]) -> pd.DataFrame:
    now_ms = int(pd.Timestamp.utcnow().tz_localize("UTC").value // 1_000_000)
    mask = df["ts"] > (now_ms - tf_ms)
    if mask.any():
        for ts in df.loc[mask, "ts"].tolist():
            _append_issue(issues, ts, "R02_TS_FUTURE")
        df = df.loc[~mask].copy()
    return df


def _rule_R03_TS_MISALIGNED(df: pd.DataFrame, tf_ms: int, tol_s: int, issues: List[Dict[str, Any]]) -> pd.DataFrame:
    off = (df["ts"] % tf_ms).astype("int64")
    mis = off != 0
    if not mis.any():
        return df
    near = (off <= tol_s * 1000) | ((tf_ms - off) <= tol_s * 1000)
    fixable = mis & near
    hard = mis & (~near)
    if fixable.any():
        fixed = ((df.loc[fixable, "ts"] // tf_ms) * tf_ms).astype("int64")
        df.loc[fixable, "ts"] = fixed
        for ts in fixed.tolist():
            _append_issue(issues, int(ts), "R03_TS_MISALIGNED", note="rounded")
    if hard.any():
        for ts in df.loc[hard, "ts"].tolist():
            _append_issue(issues, int(ts), "R03_TS_MISALIGNED", note="dropped")
        df = df.loc[~hard].copy()
    return df.sort_values("ts").reset_index(drop=True)


def _gapify_mask(df: pd.DataFrame, mask: pd.Series, issues: List[Dict[str, Any]], code: str) -> None:
    if not mask.any():
        return
    prev = _prev_close_series(df)
    for col in ["o", "h", "l", "c"]:
        if col in df.columns:
            df.loc[mask, col] = prev.loc[mask]
    if "v" in df.columns:
        df.loc[mask, "v"] = 0.0
    if "t" in df.columns:
        df.loc[mask, "t"] = 0.0
    if "is_gap" in df.columns:
        df.loc[mask, "is_gap"] = True
    else:
        df["is_gap"] = False
        df.loc[mask, "is_gap"] = True
    for ts in df.loc[mask, "ts"].tolist():
        _append_issue(issues, int(ts), code)


def _rule_R10_NEG_PRICE(df: pd.DataFrame, issues: List[Dict[str, Any]]) -> pd.DataFrame:
    need = {"o", "h", "l", "c"}
    if not need.issubset(df.columns):
        return df
    bad = (df[["o", "h", "l", "c"]] <= 0).any(axis=1)
    _gapify_mask(df, bad, issues, code="R10_NEG_PRICE")
    return df


def _rule_R11_NEG_VOL(df: pd.DataFrame, issues: List[Dict[str, Any]]) -> pd.DataFrame:
    if "v" not in df.columns:
        return df
    bad = df["v"] < 0
    if bad.any():
        df.loc[bad, "v"] = 0.0
        for ts in df.loc[bad, "ts"].tolist():
            _append_issue(issues, int(ts), "R11_NEG_VOL")
    return df


def _rule_R12_NAN(df: pd.DataFrame, issues: List[Dict[str, Any]]) -> pd.DataFrame:
    need = {"o", "h", "l", "c"}
    if not need.issubset(df.columns):
        return df
    bad = df[["o", "h", "l", "c"]].isna().any(axis=1)
    _gapify_mask(df, bad, issues, code="R12_NAN")
    return df


def _rule_R13_OHLC_ORDER(df: pd.DataFrame, issues: List[Dict[str, Any]]) -> pd.DataFrame:
    need = {"o", "h", "l", "c"}
    if not need.issubset(df.columns):
        return df
    mx = pd.concat([df.get("h"), df.get("o"), df.get("c")], axis=1).max(axis=1)
    mn = pd.concat([df.get("l"), df.get("o"), df.get("c")], axis=1).min(axis=1)
    bad = (df["h"] < mx) | (df["l"] > mn)
    if bad.any():
        df.loc[bad, "h"] = mx.loc[bad]
        df.loc[bad, "l"] = mn.loc[bad]
        eq = bad & (df["h"] == df["l"])
        _gapify_mask(df, eq, issues, code="R13_OHLC_ORDER")
        rest = bad & (~eq)
        for ts in df.loc[rest, "ts"].tolist():
            _append_issue(issues, int(ts), "R13_OHLC_ORDER")
    return df


def _rule_R14_H_LT_L(df: pd.DataFrame, issues: List[Dict[str, Any]]) -> pd.DataFrame:
    if not {"h", "l"}.issubset(df.columns):
        return df
    bad = df["h"] < df["l"]
    if bad.any():
        tmp = df.loc[bad, "h"].copy()
        df.loc[bad, "h"] = df.loc[bad, "l"]
        df.loc[bad, "l"] = tmp
        for ts in df.loc[bad, "ts"].tolist():
            _append_issue(issues, int(ts), "R14_H_LT_L")
    return df

# =============================
# Статистические правила R20–R22
# =============================

def _safe_log(x: pd.Series) -> pd.Series:
    return np.log(x.clip(lower=1e-12))


def _rolling_z(x: pd.Series, win: int) -> pd.Series:
    mu = x.rolling(win, min_periods=max(5, win//5)).mean()
    sd = x.rolling(win, min_periods=max(5, win//5)).std(ddof=1)
    return (x - mu) / sd.replace(0.0, np.nan)


def _rule_R20_VOL_SPIKE(df: pd.DataFrame, issues: List[Dict[str, Any]], win: int, z_thr: float) -> None:
    if "v" not in df.columns:
        return
    z = _rolling_z(df["v"].fillna(0.0), win)
    mask = z.abs() > z_thr
    for ts in df.loc[mask, "ts"].tolist():
        _append_issue(issues, int(ts), "R20_VOL_SPIKE")


def _atr_series(df: pd.DataFrame) -> pd.Series:
    # классический ATR на основе TR
    if not {"h", "l", "c"}.issubset(df.columns):
        return pd.Series(np.nan, index=df.index)
    h, l, c = df["h"], df["l"], df["c"].shift(1)
    tr = pd.concat([(h - l).abs(), (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr


def _rule_R21_ATR_SPIKE(df: pd.DataFrame, issues: List[Dict[str, Any]], win: int, z_thr: float) -> None:
    tr = _atr_series(df)
    z = _rolling_z(tr.fillna(0.0), win)
    mask = z.abs() > z_thr
    for ts in df.loc[mask, "ts"].tolist():
        _append_issue(issues, int(ts), "R21_ATR_SPIKE")


def _rule_R22_RET_SPIKE(df: pd.DataFrame, issues: List[Dict[str, Any]], win: int, z_thr: float) -> None:
    if "c" not in df.columns:
        return
    r = _safe_log(df["c"]).diff()
    z = _rolling_z(r.fillna(0.0), win)
    mask = z.abs() > z_thr
    for ts in df.loc[mask, "ts"].tolist():
        _append_issue(issues, int(ts), "R22_RET_SPIKE")

# =============================
# Согласованность ресемплинга R30–R33 (при наличии ref_1m и tf != '1m')
# =============================

def _resample_1m(df1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    # индекс ts msec → DatetimeIndex UTC
    ts = pd.to_datetime(df1m["ts"], unit="ms", utc=True)
    core = df1m.set_index(ts)
    freq = _to_pandas_freq(tf)
    g = core.groupby(pd.Grouper(freq=freq, label="right"))
    out = pd.DataFrame(index=g.size().index)
    if "o" in core.columns:
        out["o"] = g["o"].first()
    if "h" in core.columns:
        out["h"] = g["h"].max()
    if "l" in core.columns:
        out["l"] = g["l"].min()
    if "c" in core.columns:
        out["c"] = g["c"].last()
    if "v" in core.columns:
        out["v"] = g["v"].sum()
    if "t" in core.columns:
        out["t"] = g["t"].sum()
    out.index = (out.index.view("int64") // 1_000_000).astype("int64")  # правые границы
    out = out.reset_index().rename(columns={"index": "ts"})
    return out.dropna(how="all").reset_index(drop=True)


def _rule_R30_R31_R32_R33(
    df_tf: pd.DataFrame,
    ref_1m: pd.DataFrame,
    tf: str,
    issues: List[Dict[str, Any]],
    eps: float,
) -> None:
    # Аггрегируем 1m до tf и сравниваем
    agg = _resample_1m(ref_1m, tf)
    # левое соединение по ts
    on = sorted(set(["ts"]) | set([c for c in ["o","h","l","c","v","t"] if c in df_tf.columns and c in agg.columns]))
    j = pd.merge(df_tf[on], agg[on], on="ts", how="left", suffixes=("", "_ref"))

    def _cmp(col: str, code: str) -> None:
        if col not in j.columns or f"{col}_ref" not in j.columns:
            return
        bad = _rel_ne(j[col].astype(float), j[f"{col}_ref"].astype(float), eps)
        for ts in j.loc[bad, "ts"].tolist():
            _append_issue(issues, int(ts), code)

    _cmp("o", "R30_OHLC_MISMATCH")
    _cmp("h", "R30_OHLC_MISMATCH")
    _cmp("l", "R30_OHLC_MISMATCH")
    _cmp("c", "R30_OHLC_MISMATCH")
    _cmp("v", "R31_VOL_MISMATCH")

    # R32: High/Low вне диапазона минут
    if {"h","l"}.issubset(j.columns) and {"h_ref","l_ref"}.issubset(j.columns):
        bad_hi = j["h"] > (j["h_ref"] + np.maximum(1.0, np.abs(j["h_ref"])) * eps)
        bad_lo = j["l"] < (j["l_ref"] - np.maximum(1.0, np.abs(j["l_ref"])) * eps)
        for ts in j.loc[bad_hi | bad_lo, "ts"].tolist():
            _append_issue(issues, int(ts), "R32_RANGE_MISMATCH")

    # R33: число минут в окне
    nmin = _tf_ms(tf) // 60_000
    cnt = (
        ref_1m.assign(_ones=1)
        .groupby(pd.to_datetime(ref_1m["ts"], unit="ms", utc=True).dt.floor(_to_pandas_freq(tf)) + pd.to_timedelta(nmin, unit="m"))
        ["_ones"].sum()
    )
    cnt.index = (cnt.index.view("int64") // 1_000_000).astype("int64")
    j2 = pd.merge(df_tf[["ts"]], cnt.rename("count").reset_index().rename(columns={"index":"ts"}), on="ts", how="left")
    bad_cnt = j2["count"].fillna(0) < nmin
    for ts in j2.loc[bad_cnt, "ts"].tolist():
        _append_issue(issues, int(ts), "R33_COUNT_MISMATCH")

# =============================
# Основная точка входа
# =============================

def validate(
    df: pd.DataFrame,
    *,
    tf: str,
    symbol: Optional[str] = None,
    repair: bool = True,
    config: Optional[QualityConfig] = None,
    ref_1m: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Валидация данных C2. Возвращает (санитайзнутый df, issues_df).
    Совместимая сигнатура: validate(df, tf, symbol, repair, config, ref_1m).
    """
    cfg = config or QualityConfig()
    tf_ms = _tf_ms(tf)

    work = _ensure_ts(df)
    issues: List[Dict[str, Any]] = []

    # Время/структура
    work = _rule_R01_DUP_TS(work, issues)
    work = _rule_R02_TS_FUTURE(work, tf_ms, issues)
    work = _rule_R03_TS_MISALIGNED(work, tf_ms, cfg.misaligned_tolerance_seconds, issues)

    # Инварианты
    work = _rule_R10_NEG_PRICE(work, issues)
    work = _rule_R11_NEG_VOL(work, issues)
    work = _rule_R12_NAN(work, issues)
    work = _rule_R13_OHLC_ORDER(work, issues)
    work = _rule_R14_H_LT_L(work, issues)

    # Статистика
    _rule_R20_VOL_SPIKE(work, issues, cfg.vol_window, cfg.z_thr_vol)
    _rule_R21_ATR_SPIKE(work, issues, cfg.atr_window, cfg.z_thr_atr)
    _rule_R22_RET_SPIKE(work, issues, cfg.ret_window, cfg.z_thr_ret)

    # Согласованность TF против 1m
    if ref_1m is not None and tf != "1m":
        ref = _ensure_ts(ref_1m)
        _rule_R30_R31_R32_R33(work, ref, tf, issues, cfg.epsilon_rel)

    # Итоговый df: канонический порядок
    cols = [c for c in ["ts", "o", "h", "l", "c", "v", "t", "is_gap"] if c in work.columns]
    work = work.sort_values("ts").reset_index(drop=True)[cols]

    # Issues → DataFrame + нормализация
    issues_df = pd.DataFrame(issues, columns=["ts", "code", "note"]).drop_duplicates()
    if len(issues_df):
        if symbol is not None:
            issues_df["symbol"] = symbol
        if tf is not None:
            issues_df["tf"] = tf
        issues_df = normalize_issues_df(issues_df)
    else:
        issues_df = normalize_issues_df(pd.DataFrame(columns=["ts", "code", "note", "symbol", "tf"]))

    return work, issues_df
