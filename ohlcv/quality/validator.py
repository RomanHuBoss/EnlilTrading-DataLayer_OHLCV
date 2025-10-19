from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================
# Битовая маска dq_flags + синонимы для обратной совместимости
# =============================
DQ_BITS: Dict[str, int] = {
    # Базовые
    "INV_OHLC": 0,
    "inv_ohlc": 0,
    "MISSING_BARS": 1,
    "gap": 1,
    "MISSING": 1,  # совместимость с тестами/наследием
    "missing": 1,
    "NEG_V": 2,
    "neg_v": 2,
    "MISALIGNED_TS": 3,
    "misaligned_ts": 3,
    "MISSING_FILLED": 4,
    "missing_filled": 4,
    # Новые для C2
    "R20_VOL_SPIKE": 5,
    "R21_ATR_SPIKE": 6,
    "R22_RET_SPIKE": 7,
    "R30_OHLC_MISMATCH": 8,
    "R31_VOL_MISMATCH": 9,
    "R33_COUNT_MISMATCH": 10,
}

BIT_INV_OHLC = DQ_BITS["INV_OHLC"]
BIT_GAP = DQ_BITS["MISSING_BARS"]
BIT_NEG_V = DQ_BITS["NEG_V"]
BIT_MISALIGNED = DQ_BITS["MISALIGNED_TS"]
BIT_MISSING_FILL = DQ_BITS["MISSING_FILLED"]
BIT_R20 = DQ_BITS["R20_VOL_SPIKE"]
BIT_R21 = DQ_BITS["R21_ATR_SPIKE"]
BIT_R22 = DQ_BITS["R22_RET_SPIKE"]
BIT_R30 = DQ_BITS["R30_OHLC_MISMATCH"]
BIT_R31 = DQ_BITS["R31_VOL_MISMATCH"]
BIT_R33 = DQ_BITS["R33_COUNT_MISMATCH"]

FLAG_TO_NOTE: Dict[int, str] = {
    BIT_INV_OHLC: "inv_ohlc",
    BIT_GAP: "gap",
    BIT_NEG_V: "neg_v",
    BIT_MISALIGNED: "misaligned_ts",
    BIT_MISSING_FILL: "missing_filled",
    BIT_R20: "vol_spike",
    BIT_R21: "atr_spike",
    BIT_R22: "ret_spike",
    BIT_R30: "agg_ohlc_mismatch",
    BIT_R31: "agg_vol_mismatch",
    BIT_R33: "agg_count_mismatch",
}

# =============================
# Конфигурация
# =============================
@dataclass
class QualityConfig:
    # Порог доли пропусков для автозаполнения внутренних минут (только 1m)
    missing_fill_threshold: float = 0.0001
    # Допустимое отклонение (секунд) до правой границы минуты
    misaligned_tolerance_seconds: int = 1
    # Окна и пороги статистических правил R20–R22
    atr_window: int = 14
    vol_window: int = 50
    ret_window: int = 50
    # Метод и пороги: z-score или квантили
    use_zscore: bool = True
    z_thr_vol: float = 6.0
    z_thr_atr: float = 5.0
    z_thr_ret: float = 6.0
    q_hi_vol: float = 0.999
    q_hi_atr: float = 0.995
    q_hi_ret: float = 0.999
    # Минимально необходимая история для расчётов
    min_hist: int = 200
    # Допуск сравнения при сверке агрегатов
    compare_rtol: float = 1e-9
    compare_atol: float = 1e-8


# =============================
# Утилиты индекса/времени
# =============================

def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("ожидается DatetimeIndex")
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out = df.copy()
    out.index = idx
    out = out.sort_index()
    return out


def _align_to_right_boundary(
    df: pd.DataFrame, *, tf: str, tol_sec: int
) -> Tuple[pd.DataFrame, np.ndarray]:
    if df.empty or tf != "1m":
        return df, np.zeros(len(df), dtype=bool)
    right = df.index.ceil("min")
    delta_i8 = right.view("i8") - df.index.view("i8")
    need = np.abs(delta_i8 // 1_000_000_000) > tol_sec
    if not need.any():
        return df, need
    new_i8 = np.where(need, right.view("i8"), df.index.view("i8"))
    out = df.copy()
    out.index = pd.DatetimeIndex(new_i8.astype("datetime64[ns]")).tz_localize("UTC")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out, need


def _fill_small_internal_gaps(
    df: pd.DataFrame, *, threshold: float
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Заполнение внутренних минут при малой доле пропусков.
    Возвращает (df_на_полном_календаре, mask_is_gap).
    """
    if df.empty:
        return df, np.zeros(0, dtype=bool)
    start, end = df.index.min(), df.index.max()
    full = pd.date_range(start, end, freq="min", tz="UTC")
    if len(full) == len(df):
        out = df.copy()
        if "is_gap" not in out.columns:
            out["is_gap"] = False
        return out, out["is_gap"].to_numpy(dtype=bool)

    miss_rate = 1.0 - (len(df) / len(full))
    if miss_rate > threshold:
        out = df.copy()
        if "is_gap" not in out.columns:
            out["is_gap"] = False
        return out, out["is_gap"].to_numpy(dtype=bool)

    cols = [c for c in ["o", "h", "l", "c", "v", "t"] if c in df.columns]
    base = df[cols].reindex(full)
    prev_c = base["c"].ffill()
    syn = base[base["o"].isna()].copy()
    if not syn.empty:
        syn["o"] = prev_c.loc[syn.index]
        syn["h"] = prev_c.loc[syn.index]
        syn["l"] = prev_c.loc[syn.index]
        syn["c"] = prev_c.loc[syn.index]
        if "v" in syn.columns:
            syn["v"] = 0.0
        if "t" in syn.columns:
            syn["t"] = 0.0
        base.update(syn)
    base["is_gap"] = False
    base.loc[syn.index, "is_gap"] = True
    return base, base["is_gap"].to_numpy(dtype=bool)


# =============================
# Правки/правила на уровне одного ТФ
# =============================

def _apply_rules_basic(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]], np.ndarray]:
    out = df.copy()
    n = len(out)
    flags = np.zeros(n, dtype=np.int32)
    issues: List[Dict[str, Any]] = []

    # R12: отрицательный объём → 0
    if "v" in out.columns:
        neg = out["v"].to_numpy() < 0.0
        if neg.any():
            out.loc[out.index[neg], "v"] = 0.0
            flags[neg] |= 1 << BIT_NEG_V
            for ts in out.index[neg]:
                issues.append({
                    "ts": ts,
                    "code": "NEG_V",
                    "note": "negative volume -> 0",
                    "severity": "medium",
                    "action": "clip_to_zero",
                    "dq_rank": 30,
                })

    # R10–R11: инварианты OHLC
    o_arr = out["o"].to_numpy()
    h_arr = out["h"].to_numpy()
    l_arr = out["l"].to_numpy()
    c_arr = out["c"].to_numpy()
    inv = (h_arr < np.maximum(o_arr, c_arr)) | (l_arr > np.minimum(o_arr, c_arr))
    if inv.any():
        h_fix = np.maximum(h_arr, np.maximum(o_arr, c_arr))
        l_fix = np.minimum(l_arr, np.minimum(o_arr, c_arr))
        out.loc[out.index[inv], "h"] = h_fix[inv]
        out.loc[out.index[inv], "l"] = l_fix[inv]
        flags[inv] |= 1 << BIT_INV_OHLC
        for ts in out.index[inv]:
            issues.append({
                "ts": ts,
                "code": "INV_OHLC",
                "note": "OHLC invariant fixed",
                "severity": "high",
                "action": "fix_bounds",
                "dq_rank": 40,
            })

    return out, issues, flags


# =============================
# Статистика: TR/ATR, доходности, Z-score/квантили
# =============================

def _true_range(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr1 = h - l
    tr2 = np.abs(h - prev_c)
    tr3 = np.abs(l - prev_c)
    return np.maximum(tr1, np.maximum(tr2, tr3))


def _rolling_z(x: pd.Series, win: int) -> pd.Series:
    m = x.rolling(win, min_periods=win).mean()
    s = x.rolling(win, min_periods=win).std(ddof=0)
    return (x - m) / s


def _flag_spikes(
    df: pd.DataFrame,
    *,
    cfg: QualityConfig,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    n = len(df)
    flags = np.zeros(n, dtype=np.int32)
    issues: List[Dict[str, Any]] = []

    if n < cfg.min_hist:
        return issues, flags

    # Доходности
    ret = df["c"].pct_change().replace([np.inf, -np.inf], np.nan)

    # ATR
    tr = pd.Series(_true_range(df["o"].to_numpy(), df["h"].to_numpy(), df["l"].to_numpy(), df["c"].to_numpy()), index=df.index)
    atr = tr.rolling(cfg.atr_window, min_periods=cfg.atr_window).mean()

    # Объём
    vol = df["v"] if "v" in df.columns else pd.Series(np.nan, index=df.index)

    # Z-score / квантили
    if cfg.use_zscore:
        z_ret = _rolling_z(ret.abs(), cfg.ret_window)
        z_atr = _rolling_z(atr, cfg.vol_window)
        z_vol = _rolling_z(vol, cfg.vol_window)

        m_ret = z_ret > cfg.z_thr_ret
        m_atr = z_atr > cfg.z_thr_atr
        m_vol = z_vol > cfg.z_thr_vol
    else:
        q_ret = ret.abs().rolling(cfg.ret_window, min_periods=cfg.ret_window).quantile(cfg.q_hi_ret)
        q_atr = atr.rolling(cfg.vol_window, min_periods=cfg.vol_window).quantile(cfg.q_hi_atr)
        q_vol = vol.rolling(cfg.vol_window, min_periods=cfg.vol_window).quantile(cfg.q_hi_vol)
        m_ret = ret.abs() > q_ret
        m_atr = atr > q_atr
        m_vol = vol > q_vol

    # Проставление флагов и issues
    if m_vol.any():
        idx = df.index[m_vol.fillna(False)]
        flags[m_vol.fillna(False).to_numpy()] |= 1 << BIT_R20
        for ts in idx:
            issues.append({
                "ts": ts,
                "code": "R20_VOL_SPIKE",
                "note": "volume spike",
                "severity": "low",
                "action": "flag",
                "dq_rank": 10,
            })

    if m_atr.any():
        idx = df.index[m_atr.fillna(False)]
        flags[m_atr.fillna(False).to_numpy()] |= 1 << BIT_R21
        for ts in idx:
            issues.append({
                "ts": ts,
                "code": "R21_ATR_SPIKE",
                "note": "atr spike",
                "severity": "medium",
                "action": "flag",
                "dq_rank": 20,
            })

    if m_ret.any():
        idx = df.index[m_ret.fillna(False)]
        flags[m_ret.fillna(False).to_numpy()] |= 1 << BIT_R22
        for ts in idx:
            issues.append({
                "ts": ts,
                "code": "R22_RET_SPIKE",
                "note": "return spike",
                "severity": "medium",
                "action": "flag",
                "dq_rank": 20,
            })

    return issues, flags


# =============================
# Сверка согласованности агрегатов R30–R33
# =============================

def _resample_1m_to(df_1m: pd.DataFrame, dst_tf: str) -> pd.DataFrame:
    if df_1m.empty:
        return df_1m.copy()
    rule = {"5m": "5min", "15m": "15min", "1h": "1h"}.get(dst_tf)
    if rule is None:
        raise ValueError("dst_tf должен быть в {5m,15m,1h}")

    grp = df_1m.resample(rule, label="right", closed="right")
    o = grp["o"].first()
    h = grp["h"].max()
    l = grp["l"].min()
    c = grp["c"].last()
    v = grp["v"].sum() if "v" in df_1m.columns else None

    out = pd.concat([o.rename("o"), h.rename("h"), l.rename("l"), c.rename("c")], axis=1)
    if v is not None:
        out["v"] = v
    return out.dropna(how="all")


def _compare_frames(a: pd.DataFrame, b: pd.DataFrame, *, rtol: float, atol: float) -> Tuple[pd.Index, pd.Index]:
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return pd.Index([]), pd.Index([])
    a1, b1 = a.loc[common], b.loc[common]
    cols = [c for c in ["o", "h", "l", "c"] if c in a1.columns and c in b1.columns]
    bad_rows_mask = np.zeros(len(common), dtype=bool)
    if cols:
        diff = [~np.isclose(a1[c].to_numpy(), b1[c].to_numpy(), rtol=rtol, atol=atol) for c in cols]
        bad_rows_mask |= np.any(np.vstack(diff), axis=0)
    if "v" in a1.columns and "v" in b1.columns:
        bad_v = ~np.isclose((a1["v"].to_numpy()), (b1["v"].to_numpy()), rtol=rtol, atol=atol)
        bad_rows_mask |= bad_v
    bad_idx = common[bad_rows_mask]

    # Проверка количества баров
    count_bad = pd.Index([])
    if len(a.index) != len(b.index):
        # Найдём индексы, которых не хватает в b
        miss = a.index.difference(b.index)
        if len(miss) > 0:
            count_bad = miss
    return bad_idx, count_bad


def _consistency_checks(
    df: pd.DataFrame,
    *,
    tf: str,
    ref_1m: Optional[pd.DataFrame],
    cfg: QualityConfig,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    issues: List[Dict[str, Any]] = []
    flags = np.zeros(len(df), dtype=np.int32)

    if tf not in {"5m", "15m", "1h"}:
        return issues, flags
    if ref_1m is None or ref_1m.empty:
        return issues, flags

    a = df[[c for c in ["o", "h", "l", "c", "v"] if c in df.columns]].copy()
    b = _resample_1m_to(ref_1m[[c for c in ["o", "h", "l", "c", "v"] if c in ref_1m.columns]], tf)

    bad, count_bad = _compare_frames(a, b, rtol=cfg.compare_rtol, atol=cfg.compare_atol)

    if len(bad) > 0:
        mask = df.index.isin(bad)
        flags[mask] |= 1 << BIT_R30
        for ts in bad:
            issues.append({
                "ts": ts,
                "code": "R30_OHLC_MISMATCH",
                "note": "aggregated OHLC/V differs from 1m resample",
                "severity": "high",
                "action": "rebuild_from_1m",
                "dq_rank": 90,
            })

    if len(count_bad) > 0:
        mask = df.index.isin(count_bad)
        flags[mask] |= 1 << BIT_R33
        for ts in count_bad:
            issues.append({
                "ts": ts,
                "code": "R33_COUNT_MISMATCH",
                "note": "missing aggregated bar vs 1m calendar",
                "severity": "high",
                "action": "reindex_and_fill",
                "dq_rank": 80,
            })

    # Дополнительно: только по объёму (если OHLC совпали, а v — нет)
    if not a.empty and "v" in a.columns and "v" in b.columns:
        common = a.index.intersection(b.index)
        if len(common):
            v_bad = ~np.isclose(a.loc[common, "v"].to_numpy(), b.loc[common, "v"].to_numpy(), rtol=cfg.compare_rtol, atol=cfg.compare_atol)
            if v_bad.any():
                ts_bad = common[v_bad]
                mask = df.index.isin(ts_bad)
                flags[mask] |= 1 << BIT_R31
                for ts in ts_bad:
                    issues.append({
                        "ts": ts,
                        "code": "R31_VOL_MISMATCH",
                        "note": "aggregated volume differs from 1m sum",
                        "severity": "medium",
                        "action": "rebuild_from_1m",
                        "dq_rank": 70,
                    })
    return issues, flags


# =============================
# Формирование dq_notes и итоговая валидация
# =============================

def _notes_from_flags(flags: np.ndarray) -> List[str]:
    notes: List[str] = []
    for v in flags.tolist():
        tags = [name for bit, name in FLAG_TO_NOTE.items() if (v & (1 << bit))]
        notes.append(";".join(tags))
    return notes


def validate(
    df: pd.DataFrame,
    *,
    tf: str = "1m",
    symbol: Optional[str] = None,
    repair: bool = True,
    config: Optional[QualityConfig] = None,
    ref_1m: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Валидация качества для C2.

    Возвращает (df_out, issues_df).
    df_out содержит dq_flags, dq_notes и все правки.

    Параметры:
      tf: таймфрейм входного df ("1m"|"5m"|"15m"|"1h").
      symbol: тикер для включения в issues.
      repair: применять безопасные правки (OHLC-границы, neg_v→0, вставка синтетик при малых пропусках для 1m).
      ref_1m: эталонный 1m для сверок R30–R33 (для tf∈{5m,15m,1h}).
    """
    cfg = config or QualityConfig()
    out = _ensure_utc_index(df)

    # Выравнивание к правой границе минуты (только 1m)
    out, mis_mask_old = _align_to_right_boundary(out, tf=tf, tol_sec=cfg.misaligned_tolerance_seconds)

    # Заполнение небольших внутренних разрывов (только 1m)
    gap_mask_new = np.zeros(len(out), dtype=bool)
    if tf == "1m" and repair:
        out, gap_mask_new = _fill_small_internal_gaps(out, threshold=cfg.missing_fill_threshold)

    # Базовые правки
    out, issues, flags = _apply_rules_basic(out if repair else out.copy())

    # MISALIGNED_TS: отметка тех, кто реально сдвинулся
    if len(mis_mask_old) == len(df) and mis_mask_old.any():
        aligned_right = df.index.ceil("min")
        mis_right = aligned_right[mis_mask_old]
        aligned_pos_mask = out.index.isin(mis_right)
        if aligned_pos_mask.any():
            flags = (flags | np.where(aligned_pos_mask, (1 << BIT_MISALIGNED), 0)).astype(np.int32)
            for ts in out.index[aligned_pos_mask]:
                issues.append({
                    "ts": ts,
                    "code": "MISALIGNED_TS",
                    "note": "aligned to right boundary",
                    "severity": "low",
                    "action": "align_right",
                    "dq_rank": 5,
                })

    # Флаги и issues по вставленным барам
    if "is_gap" in out.columns:
        gap_arr = out["is_gap"].to_numpy(dtype=bool)
        if gap_arr.any():
            flags = (flags | np.where(gap_arr, (1 << BIT_GAP), 0)).astype(np.int32)
            for ts in out.index[gap_arr]:
                issues.append({
                    "ts": ts,
                    "code": "MISSING",
                    "note": "synthetic minute inserted",
                    "severity": "low",
                    "action": "synthetic_flat_bar",
                    "dq_rank": 1,
                })
        if gap_mask_new.shape == flags.shape and gap_mask_new.any():
            flags = (flags | np.where(gap_mask_new, (1 << BIT_MISSING_FILL), 0)).astype(np.int32)
            for ts in out.index[gap_mask_new]:
                issues.append({
                    "ts": ts,
                    "code": "MISSING_FILLED",
                    "note": "gap filled by synthesizer",
                    "severity": "low",
                    "action": "synthetic_flat_bar",
                    "dq_rank": 2,
                })

    # Статистические флаги R20–R22
    stat_issues, stat_flags = _flag_spikes(out, cfg=cfg)
    if stat_issues:
        issues.extend(stat_issues)
    if len(stat_flags) == len(flags):
        flags |= stat_flags.astype(np.int32)

    # Согласованность агрегатов R30–R33
    cons_issues, cons_flags = _consistency_checks(out, tf=tf, ref_1m=ref_1m, cfg=cfg)
    if cons_issues:
        issues.extend(cons_issues)
    if len(cons_flags) == len(flags):
        flags |= cons_flags.astype(np.int32)

    # Итоговые заметки
    out["dq_flags"] = flags
    out["dq_notes"] = _notes_from_flags(flags)

    # Обогащение issues служебными полями, сортировка
    if issues:
        for rec in issues:
            if symbol is not None:
                rec.setdefault("symbol", symbol)
            rec.setdefault("tf", tf)
        issues_df = pd.DataFrame(issues, columns=[
            "ts", "code", "note", "severity", "action", "dq_rank", "symbol", "tf"
        ])
        issues_df = issues_df.sort_values(["ts", "dq_rank", "code"]).reset_index(drop=True)
    else:
        issues_df = pd.DataFrame(columns=[
            "ts", "code", "note", "severity", "action", "dq_rank", "symbol", "tf"
        ])

    return out, issues_df
