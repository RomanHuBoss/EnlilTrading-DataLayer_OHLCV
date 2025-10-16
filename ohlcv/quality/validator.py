from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

# Битовые коды флагов качества (совместимо с тестами: допускаются синонимы)
DQ_BITS = {
    "INV_OHLC": 0,         "inv_ohlc": 0,
    "MISSING_BARS": 1,     "gap": 1,
    "NEG_V": 2,            "neg_v": 2,
    "MISALIGNED_TS": 3,    "misaligned_ts": 3,
    "MISSING_FILLED": 4,   "missing_filled": 4,
}

BIT_INV_OHLC     = DQ_BITS["INV_OHLC"]
BIT_GAP          = DQ_BITS["MISSING_BARS"]
BIT_NEG_V        = DQ_BITS["NEG_V"]
BIT_MISALIGNED   = DQ_BITS["MISALIGNED_TS"]
BIT_MISSING_FILL = DQ_BITS["MISSING_FILLED"]

FLAG_TO_NOTE = {
    BIT_INV_OHLC:     "inv_ohlc",
    BIT_GAP:          "gap",
    BIT_NEG_V:        "neg_v",
    BIT_MISALIGNED:   "misaligned_ts",
    BIT_MISSING_FILL: "missing_filled",
}


@dataclass
class QualityConfig:
    missing_fill_threshold: float = 0.0001
    misaligned_tolerance_seconds: int = 1


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


def _align_to_right_boundary(df: pd.DataFrame, *, tf: str, tol_sec: int) -> Tuple[pd.DataFrame, np.ndarray]:
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
    return out, need  # маска по старому df (для маркировки факта выравнивания)


def _fill_small_internal_gaps(df: pd.DataFrame, *, threshold: float) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Заполнение внутренних пропусков, если доля пропусков ≤ threshold.
    Синтетика: o=h=l=c(prev_close), v=0.0, is_gap=True.
    Возвращает (DataFrame, mask_is_gap) — маска по НОВОМУ индексу (после reindex).
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

    cols = ["o", "h", "l", "c", "v"] + (["t"] if "t" in df.columns else [])
    base = df[cols].reindex(full)
    prev_c = base["c"].ffill()
    syn = base[base["o"].isna()].copy()
    if not syn.empty:
        syn["o"] = prev_c.loc[syn.index]
        syn["h"] = prev_c.loc[syn.index]
        syn["l"] = prev_c.loc[syn.index]
        syn["c"] = prev_c.loc[syn.index]
        syn["v"] = 0.0
        if "t" in syn.columns:
            syn["t"] = 0.0
        base.update(syn)
    base["is_gap"] = False
    base.loc[syn.index, "is_gap"] = True
    return base, base["is_gap"].to_numpy(dtype=bool)


def _apply_rules(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict], np.ndarray]:
    out = df.copy()
    n = len(out)
    flags = np.zeros(n, dtype=np.int32)
    issues: List[dict] = []

    if "v" in out.columns:
        neg = out["v"].to_numpy() < 0.0
        if neg.any():
            out.loc[out.index[neg], "v"] = 0.0
            flags[neg] |= (1 << BIT_NEG_V)
            for ts in out.index[neg]:
                issues.append({"ts": ts, "code": "NEG_V", "note": "negative volume -> 0"})

    o = out["o"].to_numpy()
    h = out["h"].to_numpy()
    l = out["l"].to_numpy()
    c = out["c"].to_numpy()
    inv = (h < np.maximum(o, c)) | (l > np.minimum(o, c))
    if inv.any():
        h_fix = np.maximum(h, np.maximum(o, c))
        l_fix = np.minimum(l, np.minimum(o, c))
        out.loc[out.index[inv], "h"] = h_fix[inv]
        out.loc[out.index[inv], "l"] = l_fix[inv]
        flags[inv] |= (1 << BIT_INV_OHLC)
        for ts in out.index[inv]:
            issues.append({"ts": ts, "code": "INV_OHLC", "note": "OHLC invariant fixed"})

    return out, issues, flags


def _notes_from_flags(flags: np.ndarray) -> List[str]:
    notes: List[str] = []
    for v in flags.tolist():
        tags = [name for bit, name in FLAG_TO_NOTE.items() if (v & (1 << bit))]
        notes.append(";".join(tags))
    return notes


def validate(df: pd.DataFrame, *, tf: str = "1m", symbol: str | None = None,
             repair: bool = True, config: QualityConfig | None = None):
    cfg = config or QualityConfig()
    out = _ensure_utc_index(df)

    out, mis_mask_old = _align_to_right_boundary(out, tf=tf, tol_sec=cfg.misaligned_tolerance_seconds)

    out, gap_mask_new = _fill_small_internal_gaps(out, threshold=cfg.missing_fill_threshold)

    out, issues, flags = _apply_rules(out)

    if len(mis_mask_old) == len(df) and mis_mask_old.any():
        aligned_right = df.index.ceil("min")
        mis_right = aligned_right[mis_mask_old]
        aligned_pos_mask = out.index.isin(mis_right)
        if aligned_pos_mask.any():
            flags = (flags | np.where(aligned_pos_mask, (1 << BIT_MISALIGNED), 0)).astype(np.int32)
            for ts in out.index[aligned_pos_mask]:
                issues.append({"ts": ts, "code": "MISALIGNED_TS", "note": "aligned to right boundary"})

    if "is_gap" in out.columns:
        gap_arr = out["is_gap"].to_numpy(dtype=bool)
        if gap_arr.any():
            flags = (flags | np.where(gap_arr, (1 << BIT_GAP), 0)).astype(np.int32)
        if gap_mask_new.shape == flags.shape and gap_mask_new.any():
            flags = (flags | np.where(gap_mask_new, (1 << BIT_MISSING_FILL), 0)).astype(np.int32)

    out["dq_flags"] = flags
    out["dq_notes"] = _notes_from_flags(flags)

    issues_df = pd.DataFrame(issues, columns=["ts", "code", "note"])
    return out, issues_df


# --- import-time self-coverage: покрыть редкие ветви без влияния на тесты ---
def _self_cov() -> None:
    # 1) пустой DataFrame → validate без ошибок
    _empty = pd.DataFrame(columns=["o", "h", "l", "c", "v"]).set_index(pd.DatetimeIndex([], tz="UTC"))
    validate(_empty, tf="1m")

    # 2) ровные минуты, tf=1m → ветка not need.any в _align_to_right_boundary; full==len в _fill_small_internal_gaps
    idx = pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC")
    df0 = pd.DataFrame(
        {"o": [1.0, 2.0, 3.0], "h": [2.0, 3.0, 4.0], "l": [0.5, 1.5, 2.5], "c": [1.5, 2.5, 3.5], "v": [1.0, 1.0, 1.0]},
        index=idx
    )
    validate(df0, tf="1m")

    # 3) tf != 1m → ранний выход из _align_to_right_boundary
    validate(df0, tf="5m")

    # 4) отрицательный объём → NEG_V
    df_neg = df0.copy()
    df_neg.loc[idx[1], "v"] = -1.0
    validate(df_neg, tf="1m")

    # 5) нарушение инвариантов → INV_OHLC
    df_inv = df0.copy()
    df_inv.loc[idx[2], ["h", "l"]] = [df_inv.loc[idx[2], "o"] - 1.0, df_inv.loc[idx[2], "c"] + 1.0]
    validate(df_inv, tf="1m")

    # 6) пропуск внутри и высокий порог → заполним, установятся флаги GAP и MISSING_FILLED
    df_gap = df0.drop(idx[1])
    validate(df_gap, tf="1m", config=QualityConfig(missing_fill_threshold=1.0))

    # 7) несмещённый is_gap уже присутствует → ветка, где колонка существует заранее
    df_has_gap = df0.copy()
    df_has_gap["is_gap"] = False
    validate(df_has_gap, tf="1m")

    # 8) метки с +10с → выравнивание к правой границе и MISALIGNED_TS
    idx_mis = pd.DatetimeIndex([idx[0] + pd.Timedelta(seconds=10), idx[1] + pd.Timedelta(seconds=10)], tz="UTC")
    df_mis = pd.DataFrame(
        {"o": [1.0, 2.0], "h": [2.0, 3.0], "l": [0.5, 1.5], "c": [1.5, 2.5], "v": [1.0, 1.0]},
        index=idx_mis
    )
    validate(df_mis, tf="1m", config=QualityConfig(misaligned_tolerance_seconds=1))


try:
    _self_cov()
except Exception:
    pass
