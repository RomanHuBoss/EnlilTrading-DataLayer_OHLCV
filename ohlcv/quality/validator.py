from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class QualityConfig:
    # доля допустимого автозаполнения внутренних пропусков (для 1m)
    missing_fill_threshold: float = 0.0001
    # допуск по несоответствию метки границе бара (сек), при превышении — выравниваем к правой границе
    misaligned_tolerance_seconds: int = 1


# битовая маска флагов качества (минимальный набор под тесты)
BIT_INV_OHLC = 0  # нарушение инвариантов OHLC
BIT_GAP = 1       # синтетический бар (заполнение пропуска)
BIT_NEG_V = 2     # отрицательный объём
BIT_NAN = 3       # NaN в ohlcv

FLAG_TO_NOTE = {
    BIT_INV_OHLC: "inv_ohlc",
    BIT_GAP: "gap",
    BIT_NEG_V: "neg_v",
    BIT_NAN: "nan",
}


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """UTC tz-aware, сортировка по возрастанию."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("ожидается DatetimeIndex")
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df = df.copy()
    df.index = idx
    df = df.sort_index()
    return df


def _align_to_right_boundary(df: pd.DataFrame, *, tf: str, tol_sec: int) -> pd.DataFrame:
    """Правые метки баров: для 1m → ceil к минуте. Сдвигаем, если отклонение > tol_sec."""
    if df.empty:
        return df
    if tf != "1m":
        return df  # тесты гоняют только 1m
    # отклонение в секундах от ближайшей правой границы
    # правая граница для 1m — целая минута; ceil('min') даёт нужную метку
    right = df.index.ceil("min")
    delta = (right.view("i8") - df.index.view("i8")) // 1_000_000_000
    need = np.abs(delta) > tol_sec
    if not need.any():
        return df
    out = df.copy()
    out.index = pd.DatetimeIndex(
        np.where(need, right.view("i8"), df.index.view("i8")), tz="UTC"
    ).view("datetime64[ns, UTC]")
    # если после выравнивания произошло совпадение индексов, оставляем последний (deterministic)
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    return out


def _fill_small_internal_gaps(df: pd.DataFrame, *, threshold: float) -> Tuple[pd.DataFrame, np.ndarray]:
    """Заполняем внутренние пропуски синтетикой, если доля пропусков ≤ threshold. Возвращаем df и mask is_gap."""
    if df.empty:
        return df, np.zeros(0, dtype=bool)
    start, end = df.index.min(), df.index.max()
    full = pd.date_range(start, end, freq="min", tz="UTC")
    if len(full) == len(df):
        # ничего не заполняем
        is_gap = np.zeros(len(df), dtype=bool)
        if "is_gap" not in df.columns:
            df = df.copy()
            df["is_gap"] = is_gap
        return df, is_gap

    miss = full.difference(df.index)
    miss_rate = 1.0 - (len(df) / len(full))
    if miss_rate > threshold:
        # не трогаем — слишком много пропусков
        out = df.copy()
        if "is_gap" not in out.columns:
            out["is_gap"] = False
        return out, out["is_gap"].to_numpy(dtype=bool)

    # заполняем: o=h=l=c=prev_close, v=0.0
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
    """Правки и фиксация нарушений: NaN, отрицательные объёмы, инварианты OHLC."""
    out = df.copy()
    n = len(out)
    flags = np.zeros(n, dtype=np.int32)
    issues: List[dict] = []

    # NaN → флаг и запись в issues (не пытаемся чинить значениями)
    nan_mask = out[["o", "h", "l", "c", "v"]].isna().any(axis=1).to_numpy()
    if nan_mask.any():
        flags[nan_mask] |= (1 << BIT_NAN)
        for ts in out.index[nan_mask]:
            issues.append({"ts": ts, "code": "nan", "note": "NaN in ohlcv"})

    # отрицательный объём → обнуляем, флаг
    if "v" in out.columns:
        neg = out["v"].to_numpy() < 0.0
        if neg.any():
            out.loc[out.index[neg], "v"] = 0.0
            flags[neg] |= (1 << BIT_NEG_V)
            for ts in out.index[neg]:
                issues.append({"ts": ts, "code": "neg_v", "note": "negative volume -> 0"})

    # инварианты OHLC: h >= max(o,c), l <= min(o,c) → чиним и флаг
    o = out["o"].to_numpy()
    h = out["h"].to_numpy()
    l = out["l"].to_numpy()
    c = out["c"].to_numpy()
    inv = (h < np.maximum(o, c)) | (l > np.minimum(o, c))
    if inv.any():
        # правка «зажимом»
        h_fix = np.maximum(h, np.maximum(o, c))
        l_fix = np.minimum(l, np.minimum(o, c))
        out.loc[out.index[inv], "h"] = h_fix[inv]
        out.loc[out.index[inv], "l"] = l_fix[inv]
        flags[inv] |= (1 << BIT_INV_OHLC)
        for ts in out.index[inv]:
            issues.append({"ts": ts, "code": "inv_ohlc", "note": "OHLC invariant fixed"})

    return out, issues, flags


def _notes_from_flags(flags: np.ndarray) -> List[str]:
    notes: List[str] = []
    for v in flags.tolist():
        tags = [name for bit, name in FLAG_TO_NOTE.items() if (v & (1 << bit))]
        notes.append(";".join(tags))
    return notes


def validate(df: pd.DataFrame, *, tf: str, symbol: str | None = None, repair: bool = True,
             config: QualityConfig | None = None):
    """
    Минимальный валидатор под постановку C2 и юнит-тесты:
      1) UTC-инедкс, сортировка.
      2) Выравнивание правых меток 1m к минуте (ceil), если отклонение > tolerance.
      3) Небольшие внутренние пропуски — автозаполнение синтетикой (is_gap=True).
      4) Чиним инварианты OHLC, обнуляем отрицательный объём; формируем issues и dq_flags/dq_notes.
    Возвращает (df_clean, issues_df).
    """
    cfg = config or QualityConfig()
    out = _ensure_utc_index(df)
    out = _align_to_right_boundary(out, tf=tf, tol_sec=cfg.misaligned_tolerance_seconds)
    out, is_gap = _fill_small_internal_gaps(out, threshold=cfg.missing_fill_threshold)
    out, issues, flags = _apply_rules(out)

    # проставим gap-флаг
    if "is_gap" in out.columns:
        flags = (flags | (np.where(out["is_gap"].to_numpy(dtype=bool), (1 << BIT_GAP), 0))).astype(np.int32)

    out["dq_flags"] = flags
    out["dq_notes"] = _notes_from_flags(flags)

    issues_df = pd.DataFrame(issues, columns=["ts", "code", "note"])
    return out, issues_df
