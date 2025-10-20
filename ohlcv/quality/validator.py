# ohlcv/quality/validator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .issues import normalize_issues_df

__all__ = ["QualityConfig", "validate"]

# =============================
# Конфигурация (по C2)
# =============================
@dataclass
class QualityConfig:
    # Допуски и режим
    misaligned_tolerance_seconds: int = 1          # |misalign| ≤ 1s → округление к ближайшей границе TF
    epsilon_rel: float = 1e-6                      # относительный допуск для сравнений

    # Окна/пороги статистических правил
    vol_window: int = 50
    atr_window: int = 14
    ret_window: int = 50
    z_thr_vol: float = 6.0
    z_thr_atr: float = 6.0
    z_thr_ret: float = 6.0
    q_hi_vol: float = 0.999                        # верхний квантиль
    q_hi_atr: float = 0.999
    q_hi_ret: float = 0.999

    # R23: минимальная длина серии нулей/«плоских» баров
    zero_run_min_len: int = 5


# =============================
# Вспомогательные
# =============================

# Битовое кодирование dq_flags по правилам C2 (< 31 чтобы влезть в int32)
_CODE_TO_BIT: Dict[str, int] = {
    "R01_DUP_TS":         1 << 0,
    "R02_TS_FUTURE":      1 << 1,
    "R03_TS_MISALIGNED":  1 << 2,
    "R10_NEG_PRICE":      1 << 10,
    "R11_NEG_VOL":        1 << 11,
    "R12_NAN":            1 << 12,
    "R13_OHLC_ORDER":     1 << 13,
    "R14_H_LT_L":         1 << 14,
    "R20_VOL_SPIKE":      1 << 20,
    "R21_ATR_SPIKE":      1 << 21,
    "R22_RET_SPIKE":      1 << 22,
    "R23_ZERO_RUN":       1 << 23,
    # Согласованность TF vs 1m
    "R30_OHLC_MISMATCH":  1 << 30,
    "R31_VOL_MISMATCH":   1 << 29,
    "R32_RANGE_MISMATCH": 1 << 28,
    "R33_COUNT_MISMATCH": 1 << 27,
}


def _ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ts" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            idx: pd.DatetimeIndex = pd.DatetimeIndex(out.index)
            idx = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
            out["ts"] = (idx.asi8 // 1_000_000).astype("int64")
        else:
            raise ValueError("ожидался столбец 'ts' или DatetimeIndex")
    out["ts"] = pd.to_numeric(out["ts"], errors="coerce").astype("int64")
    for c in ("o", "h", "l", "c", "v", "t"):
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
    den = np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    return (np.abs(a - b) / den) > eps


def _rolling_quantile(x: pd.Series, n: int, q: float) -> pd.Series:
    return x.rolling(n, min_periods=max(5, n // 5)).quantile(q)

# =============================
# Правила C2 (время/структура)
# =============================

def _rule_R01_DUP_TS(df: pd.DataFrame, issues: List[Dict[str, Any]]) -> pd.DataFrame:
    dup = df["ts"].duplicated(keep="last")
    if dup.any():
        for ts in df.loc[dup, "ts"].tolist():
            _append_issue(issues, ts, "R01_DUP_TS")
        df = df.loc[~dup].copy()
    return df


def _rule_R02_TS_FUTURE(df: pd.DataFrame, tf_ms: int, issues: List[Dict[str, Any]]) -> pd.DataFrame:
    # отсечение будущих баров: ts > floor(now_utc, TF)
    now_floor_tf_ms = int((pd.Timestamp.utcnow().floor(f"{tf_ms // 60_000}min").value) // 1_000_000)
    mask = df["ts"] > now_floor_tf_ms
    if mask.any():
        for ts in df.loc[mask, "ts"].tolist():
            _append_issue(issues, ts, "R02_TS_FUTURE")
        df = df.loc[~mask].copy()
    return df


def _rule_R03_TS_MISALIGNED(df: pd.DataFrame, tf_ms: int, tol_s: int, issues: List[Dict[str, Any]], repair: bool) -> pd.DataFrame:
    off = (df["ts"] % tf_ms).astype("int64")
    mis = off != 0
    if not mis.any():
        return df
    tol_ms = int(tol_s * 1000)
    near = (off <= tol_ms) | ((tf_ms - off) <= tol_ms)
    fixable = mis & near
    hard = mis & (~near)
    if fixable.any():
        if repair:
            # округление к ближайшему кратному TF
            rounded = (((df.loc[fixable, "ts"] + tf_ms // 2) // tf_ms) * tf_ms).astype("int64")
            df.loc[fixable, "ts"] = rounded
            for ts in rounded.tolist():
                _append_issue(issues, int(ts), "R03_TS_MISALIGNED", note="rounded")
        else:
            for ts in df.loc[fixable, "ts"].tolist():
                _append_issue(issues, int(ts), "R03_TS_MISALIGNED", note="near")
    if hard.any():
        for ts in df.loc[hard, "ts"].tolist():
            _append_issue(issues, int(ts), "R03_TS_MISALIGNED", note="dropped")
        df = df.loc[~hard].copy()
    return df.sort_values("ts").reset_index(drop=True)


# =============================
# Инварианты OHLC/объём
# =============================

def _gapify_mask(df: pd.DataFrame, mask: pd.Series, issues: List[Dict[str, Any]], code: str, repair: bool) -> None:
    if not mask.any():
        return
    if repair:
        prev = _prev_close_series(df)
        for col in ("o", "h", "l", "c"):
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


def _rule_R10_NEG_PRICE(df: pd.DataFrame, issues: List[Dict[str, Any]], repair: bool) -> pd.DataFrame:
    need = {"o", "h", "l", "c"}
    if not need.issubset(df.columns):
        return df
    bad = (df[["o", "h", "l", "c"]] <= 0).any(axis=1)
    _gapify_mask(df, bad, issues, code="R10_NEG_PRICE", repair=repair)
    return df


def _rule_R11_NEG_VOL(df: pd.DataFrame, issues: List[Dict[str, Any]], repair: bool) -> pd.DataFrame:
    if "v" not in df.columns:
        return df
    bad = df["v"] < 0
    if bad.any():
        if repair:
            df.loc[bad, "v"] = 0.0
        for ts in df.loc[bad, "ts"].tolist():
            _append_issue(issues, int(ts), "R11_NEG_VOL")
    return df


def _rule_R12_NAN(df: pd.DataFrame, issues: List[Dict[str, Any]], repair: bool) -> pd.DataFrame:
    need = {"o", "h", "l", "c"}
    if not need.issubset(df.columns):
        return df
    bad = df[["o", "h", "l", "c"]].isna().any(axis=1)
    _gapify_mask(df, bad, issues, code="R12_NAN", repair=repair)
    return df


def _rule_R13_OHLC_ORDER(df: pd.DataFrame, issues: List[Dict[str, Any]], repair: bool) -> pd.DataFrame:
    need = {"o", "h", "l", "c"}
    if not need.issubset(df.columns):
        return df
    mx = pd.concat([df.get("h"), df.get("o"), df.get("c")], axis=1).max(axis=1)
    mn = pd.concat([df.get("l"), df.get("o"), df.get("c")], axis=1).min(axis=1)
    bad = (df["h"] < mx) | (df["l"] > mn)
    if bad.any():
        if repair:
            df.loc[bad, "h"] = mx.loc[bad]
            df.loc[bad, "l"] = mn.loc[bad]
            eq = bad & (df["h"] == df["l"])  # всё схлопнулось → gapify
            _gapify_mask(df, eq, issues, code="R13_OHLC_ORDER", repair=True)
            rest = bad & (~eq)
            for ts in df.loc[rest, "ts"].tolist():
                _append_issue(issues, int(ts), "R13_OHLC_ORDER")
        else:
            for ts in df.loc[bad, "ts"].tolist():
                _append_issue(issues, int(ts), "R13_OHLC_ORDER")
    return df


def _rule_R14_H_LT_L(df: pd.DataFrame, issues: List[Dict[str, Any]], repair: bool) -> pd.DataFrame:
    if not {"h", "l"}.issubset(df.columns):
        return df
    bad = df["h"] < df["l"]
    if bad.any():
        if repair:
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
    mu = x.rolling(win, min_periods=max(5, win // 5)).mean()
    sd = x.rolling(win, min_periods=max(5, win // 5)).std(ddof=1)
    return (x - mu) / sd.replace(0.0, np.nan)


def _atr_wilder(df: pd.DataFrame, n: int) -> pd.Series:
    if not {"h", "l", "c"}.issubset(df.columns):
        return pd.Series(np.nan, index=df.index)
    h, l, c = df["h"], df["l"], df["c"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / float(max(1, n)), adjust=False, min_periods=n).mean()
    return atr


def _rule_R20_VOL_SPIKE(df: pd.DataFrame, issues: List[Dict[str, Any]], win: int, z_thr: float, q_hi: float) -> None:
    if "v" not in df.columns:
        return
    v = df["v"].fillna(0.0)
    z = _rolling_z(v, win)
    q = _rolling_quantile(v, win, q_hi)
    mask = (z.abs() > z_thr) | (v > q)
    for ts in df.loc[mask.fillna(False), "ts"].tolist():
        _append_issue(issues, int(ts), "R20_VOL_SPIKE")


def _rule_R21_ATR_SPIKE(df: pd.DataFrame, issues: List[Dict[str, Any]], n: int, z_thr: float, q_hi: float) -> None:
    atr = _atr_wilder(df, n)
    z = _rolling_z(atr.fillna(0.0), max(n * 4, 50))
    q = _rolling_quantile(atr, max(n * 4, 50), q_hi)
    mask = (z.abs() > z_thr) | (atr > q)
    for ts in df.loc[mask.fillna(False), "ts"].tolist():
        _append_issue(issues, int(ts), "R21_ATR_SPIKE")


def _rule_R22_RET_SPIKE(df: pd.DataFrame, issues: List[Dict[str, Any]], win: int, z_thr: float, q_hi: float) -> None:
    if "c" not in df.columns:
        return
    r = _safe_log(df["c"]).diff().fillna(0.0).abs()
    z = _rolling_z(r, win)
    q = _rolling_quantile(r, win, q_hi)
    mask = (z.abs() > z_thr) | (r > q)
    for ts in df.loc[mask.fillna(False), "ts"].tolist():
        _append_issue(issues, int(ts), "R22_RET_SPIKE")


# =============================
# R23: Протяжённые нули (ZERO_RUN)
# =============================

def _rule_R23_ZERO_RUN(df: pd.DataFrame, issues: List[Dict[str, Any]], min_len: int) -> None:
    """
    Детект длительных «плоских» серий:
      — диапазон бара == 0 (h==l==o==c) И/ИЛИ объём == 0
      — считаем только серии длиной >= min_len
    """
    if df.empty:
        return
    rng0 = pd.Series(False, index=df.index)
    if {"h", "l", "o", "c"}.issubset(df.columns):
        rng0 = (df["h"] == df["l"]) & (df["o"] == df["c"]) & (df["h"] == df["c"])  # строго плоские свечи
    vol0 = df["v"].eq(0.0) if "v" in df.columns else pd.Series(False, index=df.index)
    flat = rng0 | vol0
    if not flat.any():
        return
    s = flat.to_numpy()
    i = 0
    n = len(s)
    while i < n:
        if s[i]:
            j = i
            while j < n and s[j]:
                j += 1
            if (j - i) >= min_len:
                for ts in df.loc[i:j - 1, "ts"].tolist():
                    _append_issue(issues, int(ts), "R23_ZERO_RUN")
            i = j
        else:
            i += 1


# =============================
# Согласованность ресемплинга R30–R33 (при наличии ref_1m и tf != '1m')
# =============================

def _resample_1m_start_ts(df1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Агрегирует 1m в tf с контрактным ts = НАЧАЛО окна."""
    tf_ms = _tf_ms(tf)
    ts = pd.to_datetime(df1m["ts"], unit="ms", utc=True)
    core = df1m.set_index(ts)
    freq = _to_pandas_freq(tf)
    g = core.groupby(pd.Grouper(freq=freq, label="right", closed="right"))
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
    right_ms = (pd.DatetimeIndex(out.index).tz_convert("UTC").asi8 // 1_000_000).astype("int64")
    start_ms = right_ms - tf_ms
    out = out.reset_index(drop=True)
    out.insert(0, "ts", start_ms)
    return out.dropna(how="all").reset_index(drop=True)


def _rule_R30_R31_R32_R33(
    df_tf: pd.DataFrame,
    ref_1m: pd.DataFrame,
    tf: str,
    issues: List[Dict[str, Any]],
    eps: float,
) -> None:
    agg = _resample_1m_start_ts(ref_1m, tf)

    cols_cmp = [c for c in ["o", "h", "l", "c", "v", "t"] if c in df_tf.columns and c in agg.columns]
    j = pd.merge(df_tf[["ts"] + cols_cmp], agg[["ts"] + cols_cmp], on="ts", how="left", suffixes=("", "_ref"))

    # R30: Open/Close mismatch сводим в общий код "OHLC_MISMATCH" (см. registry)
    if "o" in cols_cmp:
        bad_o = _rel_ne(j["o"].astype(float), j["o_ref"].astype(float), eps)
        for ts in j.loc[bad_o, "ts"].tolist():
            _append_issue(issues, int(ts), "R30_OHLC_MISMATCH")
    if "c" in cols_cmp:
        bad_c = _rel_ne(j["c"].astype(float), j["c_ref"].astype(float), eps)
        for ts in j.loc[bad_c, "ts"].tolist():
            _append_issue(issues, int(ts), "R30_OHLC_MISMATCH")

    # R32: High/Low вне диапазона минут
    if {"h", "l"}.issubset(cols_cmp):
        bad_hi = j["h"] > (j["h_ref"] + np.maximum(1.0, np.abs(j["h_ref"])) * eps)
        bad_lo = j["l"] < (j["l_ref"] - np.maximum(1.0, np.abs(j["l_ref"])) * eps)
        for ts in j.loc[(bad_hi | bad_lo), "ts"].tolist():
            _append_issue(issues, int(ts), "R32_RANGE_MISMATCH")

    # R31: Volume mismatch vs sum of minutes (ε = max(1e-6, 1e-6·sum))
    if "v" in cols_cmp:
        diff = (j["v"].astype(float) - j["v_ref"].astype(float)).abs()
        tol = np.maximum(1e-6, 1e-6 * j["v_ref"].abs())
        bad_v = diff > tol
        for ts in j.loc[bad_v, "ts"].tolist():
            _append_issue(issues, int(ts), "R31_VOL_MISMATCH")

    # R33: кол-во минут в окне
    nmin = _tf_ms(tf) // 60_000
    dt = pd.to_datetime(ref_1m["ts"], unit="ms", utc=True)
    right = dt.dt.floor(_to_pandas_freq(tf)) + pd.to_timedelta(nmin, unit="m")
    cnt = ref_1m.assign(_ones=1).groupby(right)["_ones"].sum()
    cnt.index = (pd.DatetimeIndex(cnt.index).asi8 // 1_000_000 - _tf_ms(tf)).astype("int64")
    j2 = pd.merge(df_tf[["ts"]], cnt.rename("count").reset_index().rename(columns={"index": "ts"}), on="ts", how="left")
    bad_cnt = j2["count"].fillna(0) < nmin
    for ts in j2.loc[bad_cnt, "ts"].tolist():
        _append_issue(issues, int(ts), "R33_COUNT_MISMATCH")


# =============================
# Аннотации dq_flags / dq_notes
# =============================

def _build_flags_and_lastnote(raw_issues: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    if raw_issues is None or len(raw_issues) == 0:
        return pd.Series(dtype="int64"), pd.Series(dtype="string")
    r = raw_issues.copy()
    # флаги — OR всех кодов на ts
    flags = r.groupby("ts")["code"].apply(lambda codes: int(np.bitwise_or.reduce([_CODE_TO_BIT.get(c, 0) for c in codes])))
    # последний по порядку обработки код на этом ts → dq_notes
    last_note = r.groupby("ts").tail(1).set_index("ts")["code"].astype("string")
    return flags.astype("int64"), last_note


def _attach_dq_annotations(df: pd.DataFrame, flags: pd.Series, last_note: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out = pd.merge(out, flags.rename("dq_flags"), on="ts", how="left")
    out = pd.merge(out, last_note.rename("dq_notes"), on="ts", how="left")
    out["dq_flags"] = out["dq_flags"].fillna(0).astype("int32")
    out["dq_notes"] = out["dq_notes"].fillna("").astype("string")
    return out


# =============================
# Основная точка входа
# =============================

def validate(
    df: pd.DataFrame,
    *,
    tf: str,
    symbol: Optional[str] = None,
    repair: bool = True,                           # True → repair_safe; False → strict
    config: Optional[QualityConfig] = None,
    ref_1m: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Валидация данных C2. Возвращает (санитайзнутый df, issues_df).

    Контракт C2:
      — реализованы R01–R03, R10–R14, R20–R23, R30–R33;
      — режимы strict/repair_safe через параметр repair;
      — детерминированный результат;
      — выходной df дополнен dq_flags:int32 и dq_notes:str (последний код);
      — issues_df нормализован через normalize_issues_df.
    """
    cfg = config or QualityConfig()
    tf_ms = _tf_ms(tf)

    work = _ensure_ts(df)
    issues: List[Dict[str, Any]] = []

    # Время/структура
    work = _rule_R01_DUP_TS(work, issues)
    work = _rule_R02_TS_FUTURE(work, tf_ms, issues)
    work = _rule_R03_TS_MISALIGNED(work, tf_ms, cfg.misaligned_tolerance_seconds, issues, repair=repair)

    # Инварианты
    work = _rule_R10_NEG_PRICE(work, issues, repair=repair)
    work = _rule_R11_NEG_VOL(work, issues, repair=repair)
    work = _rule_R12_NAN(work, issues, repair=repair)
    work = _rule_R13_OHLC_ORDER(work, issues, repair=repair)
    work = _rule_R14_H_LT_L(work, issues, repair=repair)

    # Статистика
    _rule_R20_VOL_SPIKE(work, issues, cfg.vol_window, cfg.z_thr_vol, cfg.q_hi_vol)
    _rule_R21_ATR_SPIKE(work, issues, cfg.atr_window, cfg.z_thr_atr, cfg.q_hi_atr)
    _rule_R22_RET_SPIKE(work, issues, cfg.ret_window, cfg.z_thr_ret, cfg.q_hi_ret)
    _rule_R23_ZERO_RUN(work, issues, cfg.zero_run_min_len)

    # Согласованность TF против 1m
    if ref_1m is not None and tf != "1m":
        ref = _ensure_ts(ref_1m)
        _rule_R30_R31_R32_R33(work, ref, tf, issues, cfg.epsilon_rel)

    # Канонический порядок колонок
    cols = [c for c in ["ts", "o", "h", "l", "c", "v", "t", "is_gap"] if c in work.columns]
    work = work.sort_values("ts").reset_index(drop=True)[cols]

    # Issues → DataFrame (сырой порядок для dq_notes) и нормализация для контракта C2
    raw_issues_df = pd.DataFrame(issues, columns=["ts", "code", "note"]).drop_duplicates()
    flags, last_note = _build_flags_and_lastnote(raw_issues_df)

    if len(raw_issues_df):
        if symbol is not None:
            raw_issues_df["symbol"] = symbol
        if tf is not None:
            raw_issues_df["tf"] = tf
        issues_df = normalize_issues_df(raw_issues_df)
    else:
        issues_df = normalize_issues_df(pd.DataFrame(columns=["ts", "code", "note", "symbol", "tf"]))

    # Прикрепление dq_* к основному df
    work = _attach_dq_annotations(work, flags, last_note)

    return work, issues_df
