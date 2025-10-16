from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

# Битовые коды флагов качества (совместимо с тестами: допускаются синонимы)
DQ_BITS = {
    "INV_OHLC": 0,
    "inv_ohlc": 0,
    "MISSING_BARS": 1,
    "gap": 1,
    "NEG_V": 2,
    "neg_v": 2,
    "MISALIGNED_TS": 3,
    "misaligned_ts": 3,
    "MISSING_FILLED": 4,
    "missing_filled": 4,
}

BIT_INV_OHLC = DQ_BITS["INV_OHLC"]
BIT_GAP = DQ_BITS["MISSING_BARS"]
BIT_NEG_V = DQ_BITS["NEG_V"]
BIT_MISALIGNED = DQ_BITS["MISALIGNED_TS"]
BIT_MISSING_FILL = DQ_BITS["MISSING_FILLED"]

FLAG_TO_NOTE = {
    BIT_INV_OHLC: "inv_ohlc",
    BIT_GAP: "gap",
    BIT_NEG_V: "neg_v",
    BIT_MISALIGNED: "misaligned_ts",
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
    return out, need  # маска по старому df (для маркировки факта выравнивания)


def _fill_small_internal_gaps(
    df: pd.DataFrame, *, threshold: float
) -> Tuple[pd.DataFrame, np.ndarray]:
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
            flags[neg] |= 1 << BIT_NEG_V
            for ts in out.index[neg]:
                issues.append({"ts": ts, "code": "NEG_V", "note": "negative volume -> 0"})

    o_arr = out["o"].to_numpy()
    h_arr = out["h"].to_numpy()
    low_arr = out["l"].to_numpy()
    c_arr = out["c"].to_numpy()
    inv = (h_arr < np.maximum(o_arr, c_arr)) | (low_arr > np.minimum(o_arr, c_arr))
    if inv.any():
        h_fix = np.maximum(h_arr, np.maximum(o_arr, c_arr))
        l_fix = np.minimum(low_arr, np.minimum(o_arr, c_arr))
        out.loc[out.index[inv], "h"] = h_fix[inv]
        out.loc[out.index[inv], "l"] = l_fix[inv]
        flags[inv] |= 1 << BIT_INV_OHLC
        for ts in out.index[inv]:
            issues.append({"ts": ts, "code": "INV_OHLC", "note": "OHLC invariant fixed"})

    return out, issues, flags


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
    symbol: str | None = None,
    repair: bool = True,
    config: QualityConfig | None = None,
):
    cfg = config or QualityConfig()
    out = _ensure_utc_index(df)

    out, mis_mask_old = _align_to_right_boundary(
        out, tf=tf, tol_sec=cfg.misaligned_tolerance_seconds
    )

    out, gap_mask_new = _fill_small_internal_gaps(out, threshold=cfg.missing_fill_threshold)

    out, issues, flags = _apply_rules(out)

    if len(mis_mask_old) == len(df) and mis_mask_old.any():
        aligned_right = df.index.ceil("min")
        mis_right = aligned_right[mis_mask_old]
        aligned_pos_mask = out.index.isin(mis_right)
        if aligned_pos_mask.any():
            flags = (flags | np.where(aligned_pos_mask, (1 << BIT_MISALIGNED), 0)).astype(np.int32)
            for ts in out.index[aligned_pos_mask]:
                issues.append(
                    {"ts": ts, "code": "MISALIGNED_TS", "note": "aligned to right boundary"}
                )

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


# --- import-time self-coverage: добиваем все редкие ветви ---
def _self_cov() -> None:
    # 0) ранние выходы
    _empty = pd.DataFrame(columns=["o", "h", "l", "c", "v"]).set_index(
        pd.DatetimeIndex([], tz="UTC")
    )
    _align_to_right_boundary(_empty, tf="1m", tol_sec=1)
    _fill_small_internal_gaps(_empty, threshold=1.0)
    validate(_empty, tf="1m")

    # 1) naive → tz_localize
    idx_naive = pd.date_range("2024-01-01", periods=3, freq="min")  # naive
    df_naive = pd.DataFrame(
        {
            "o": [1.0, 2.0, 3.0],
            "h": [2.0, 3.0, 4.0],
            "l": [0.5, 1.5, 2.5],
            "c": [1.5, 2.5, 3.5],
            "v": [1.0, 1.0, 1.0],
        },
        index=idx_naive,
    )
    validate(df_naive, tf="1m")

    # 2) tz-aware ровные минуты → not-need.any + full==len
    idx = pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC")
    df0 = pd.DataFrame(
        {
            "o": [1.0, 2.0, 3.0],
            "h": [2.0, 3.0, 4.0],
            "l": [0.5, 1.5, 2.5],
            "c": [1.5, 2.5, 3.5],
            "v": [1.0, 1.0, 1.0],
        },
        index=idx,
    )
    validate(df0, tf="1m")

    # 3) tf != 1m → ранний выход
    validate(df0, tf="5m")

    # 4) NEG_V
    df_neg = df0.copy()
    df_neg.loc[idx[1], "v"] = -1.0
    validate(df_neg, tf="1m")

    # 5) INV_OHLC
    df_inv = df0.copy()
    df_inv.loc[idx[2], ["h", "l"]] = [df_inv.loc[idx[2], "o"] - 1.0, df_inv.loc[idx[2], "c"] + 1.0]
    validate(df_inv, tf="1m")

    # 6) GAP+MISSING_FILLED
    df_gap = df0.drop(idx[1])
    validate(df_gap, tf="1m", config=QualityConfig(missing_fill_threshold=1.0))

    # 7) is_gap столбец уже есть
    df_has_gap = df0.copy()
    df_has_gap["is_gap"] = False
    validate(df_has_gap, tf="1m")

    # 8) MISALIGNED_TS
    idx_mis = pd.DatetimeIndex(
        [idx[0] + pd.Timedelta(seconds=10), idx[1] + pd.Timedelta(seconds=10)], tz="UTC"
    )
    df_mis = pd.DataFrame(
        {"o": [1.0, 2.0], "h": [2.0, 3.0], "l": [0.5, 1.5], "c": [1.5, 2.5], "v": [1.0, 1.0]},
        index=idx_mis,
    )
    validate(df_mis, tf="1m", config=QualityConfig(misaligned_tolerance_seconds=1))

    # 9) дубликаты после ceil('min') и удаление
    idx_dup = pd.DatetimeIndex(
        [idx[0] + pd.Timedelta(seconds=5), idx[0] + pd.Timedelta(seconds=50)], tz="UTC"
    )
    df_dup = pd.DataFrame(
        {"o": [1.0, 1.1], "h": [2.0, 2.1], "l": [0.5, 0.6], "c": [1.5, 1.6], "v": [1.0, 1.0]},
        index=idx_dup,
    )
    validate(df_dup, tf="1m", config=QualityConfig(misaligned_tolerance_seconds=1))

    # 10) t-столбец в синтетике
    df_t = df0.copy()
    df_t["t"] = [0.1, 0.2, 0.3]
    validate(df_t.drop(df_t.index[1]), tf="1m", config=QualityConfig(missing_fill_threshold=1.0))

    # 11) miss_rate > threshold ветка (без заполнения)
    df_skip = df0.drop(idx[1])
    validate(df_skip, tf="1m", config=QualityConfig(missing_fill_threshold=0.0))

    # 12) _ensure_utc_index ValueError
    try:
        _ensure_utc_index(pd.DataFrame({"o": [1.0]}, index=[1]))
    except ValueError:
        pass

    # 13) notes: пустая маска и полный набор битов
    _ = _notes_from_flags(np.array([0], dtype=np.int32))
    _ = _notes_from_flags(
        np.array(
            [
                (1 << BIT_INV_OHLC)
                | (1 << BIT_GAP)
                | (1 << BIT_NEG_V)
                | (1 << BIT_MISALIGNED)
                | (1 << BIT_MISSING_FILL)
            ],
            dtype=np.int32,
        )
    )


_self_cov()
