# Валидатор и санитайзер рядов OHLCV для 1m/5m/15m/1h (C2 DataQuality).
# Интерфейс: validate(df, *, tf="1m", symbol=None, repair=True) -> (df_clean, issues_df)
# Детерминируемые правки: выравнивание таймстампов к границе ТФ, устранение дублей,
# приведение типов, коррекция OHLC-инвариантов, запрет отрицательного объёма,
# допустимое заполнение пропусков для 1m. Также выставляются колонки dq_flags/dq_notes.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd

from .issues import Issue, issues_to_frame


# ---- Битовая маска dq_flags --------------------------------------------------
# Для пост-анализа по строкам. Биты не взаимоисключающие, суммируются.
DQ_BITS: Dict[str, int] = {
    "DUPLICATE": 0,
    "MISALIGNED_TS": 1,
    "NON_UTC_INDEX": 2,
    "NON_MONOTONIC": 3,
    "NON_NUMERIC": 4,
    "NAN_PRICE": 5,
    "NAN_VOLUME": 6,
    "OHLC_INVARIANT": 7,
    "NEG_VOLUME": 8,
    "MISSING_BARS": 9,       # отмечаются только вставленные минуты
    "MISSING_FILLED": 10,    # синонимично MISSING_BARS для вставок
    "PRICE_SPIKE": 11,
    "LONG_ZERO_VOLUME": 12,
    "FUTURE_TS": 13,
}


def _bit(code: str) -> int:
    return 1 << DQ_BITS[code]


def _set_flag(flags: pd.Series, mask: pd.Series | np.ndarray, code: str) -> None:
    bit = _bit(code)
    flags[mask] = flags[mask].astype(np.int32) | bit


# ---- Конфигурация ------------------------------------------------------------

@dataclass
class QualityConfig:
    missing_fill_threshold: float = 0.0001  # ≤ 0.01% для 1m
    forbid_negative_volume: bool = True
    fix_ohlc_invariants: bool = True
    fix_misaligned_ts: bool = True
    misaligned_tolerance_seconds: int = 1   # допустимое отклонение до ближайшей границы
    spike_window: int = 200
    spike_k: float = 12.0
    flat_streak_threshold: int = 300


# ---- Утилиты -----------------------------------------------------------------

def _tf_freq(tf: str) -> str:
    return {"1m": "min", "5m": "5min", "15m": "15min", "1h": "1h"}[tf]


def _ensure_index(df: pd.DataFrame, issues: List[Issue], symbol: Optional[str], tf: str, cfg: QualityConfig) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Ожидается DatetimeIndex")
    # tz
    if df.index.tz is None or str(df.index.tz) != "UTC":
        issues.append(Issue(ts=pd.Timestamp.utcnow().tz_localize("UTC"), code="NON_UTC_INDEX", severity="error",
                            msg="Индекс не tz-aware UTC, приведён к UTC", symbol=symbol, tf=tf))
        df = df.tz_convert("UTC") if df.index.tz is not None else df.tz_localize("UTC")
    # сортировка
    if not df.index.is_monotonic_increasing:
        issues.append(Issue(ts=df.index.max(), code="NON_MONOTONIC", severity="warning",
                            msg="Индекс не монотонный, отсортирован", symbol=symbol, tf=tf))
        df = df.sort_index()
    # дубли
    if df.index.duplicated().any():
        n_dup = int(df.index.duplicated().sum())
        issues.append(Issue(ts=df.index.max(), code="DUPLICATE", severity="error",
                            msg=f"Дубликаты таймстампов: {n_dup}, оставлены последние", symbol=symbol, tf=tf,
                            extra={"count": n_dup}))
        df = df[~df.index.duplicated(keep="last")]
    # будущее
    now = pd.Timestamp.utcnow().tz_localize("UTC")
    fut = df.index > now
    if fut.any():
        cnt = int(fut.sum())
        issues.append(Issue(ts=df.index.max(), code="FUTURE_TS", severity="error",
                            msg=f"Бары из будущего: {cnt}, удалены", symbol=symbol, tf=tf, extra={"count": cnt}))
        df = df[~fut]
    # выравнивание к границе ТФ
    if cfg.fix_misaligned_ts and len(df) > 0:
        # допускаем отклонение ±tolerance сек от ближайшей границы; иначе — флоор к правой границе окна
        secs = (df.index.view("i8") // 10**9).astype(np.int64)
        if tf.endswith("m"):
            mod = int(tf[:-1]) * 60
            right_shift = int(tf[:-1]) * 60
        else:
            mod = 3600
            right_shift = 3600
        rem = secs % mod
        mis = (rem != 0) & (np.minimum(rem, mod - rem) > cfg.misaligned_tolerance_seconds)
        if mis.any():
            cnt = int(mis.sum())
            issues.append(Issue(ts=df.index.max(), code="MISALIGNED_TS", severity="error",
                                msg=f"Невыровненные таймстампы: {cnt}, округлены к правой границе",
                                symbol=symbol, tf=tf, extra={"count": cnt}))
            # округление вниз, затем сдвиг к правой границе окна
            floored = (secs // mod) * mod + right_shift
            new_idx = pd.to_datetime(floored, unit="s", utc=True)
            # заменяем только у неверных
            idx = df.index.to_series()
            idx.loc[mis] = new_idx[mis]
            df.index = pd.DatetimeIndex(idx.values).tz_convert("UTC")
            # возможные коллизии — оставляем последние
            df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _coerce_types(df: pd.DataFrame, issues: List[Issue], symbol: Optional[str], tf: str) -> pd.DataFrame:
    needed = ["o", "h", "l", "c", "v"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Отсутствует обязательный столбец: {col}")
    # приведение к числам
    non_numeric_cols = []
    for col in ["o", "h", "l", "c", "v"]:
        if not np.issubdtype(df[col].dtype, np.number):
            non_numeric_cols.append(col)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if non_numeric_cols:
        issues.append(Issue(ts=df.index.max(), code="NON_NUMERIC", severity="error",
                            msg=f"Приведены к float: {','.join(non_numeric_cols)}", symbol=symbol, tf=tf))
    # NaN
    nan_price_cnt = int(df[["o", "h", "l", "c"]].isna().sum().sum())
    nan_vol_cnt = int(df["v"].isna().sum())
    if nan_price_cnt:
        issues.append(Issue(ts=df.index.max(), code="NAN_PRICE", severity="error",
                            msg=f"NaN в ценах: {nan_price_cnt}", symbol=symbol, tf=tf,
                            extra={"count": nan_price_cnt}))
    if nan_vol_cnt:
        issues.append(Issue(ts=df.index.max(), code="NAN_VOLUME", severity="warning",
                            msg=f"NaN в объёме: {nan_vol_cnt}, заменены на 0",
                            symbol=symbol, tf=tf, extra={"count": nan_vol_cnt}))
        df["v"] = df["v"].fillna(0.0)
    return df


def _fix_ohlc_invariants(df: pd.DataFrame, issues: List[Issue], symbol: Optional[str], tf: str, cfg: QualityConfig,
                         dq_flags: pd.Series) -> pd.DataFrame:
    if not cfg.fix_ohlc_invariants:
        return df
    max_oc = np.fmax(df["o"].values, df["c"].values)
    min_oc = np.fmin(df["o"].values, df["c"].values)

    bad_h = df["h"].values < max_oc
    bad_l = df["l"].values > min_oc
    bad_hl = df["h"].values < df["l"].values

    n_bad = int(bad_h.sum() + bad_l.sum() + bad_hl.sum())
    if n_bad:
        issues.append(Issue(ts=df.index.max(), code="OHLC_INVARIANT", severity="error",
                            msg=f"Нарушения OHLC-инвариантов: {n_bad}, применена коррекция",
                            symbol=symbol, tf=tf,
                            extra={"bad_h": int(bad_h.sum()), "bad_l": int(bad_l.sum()), "bad_hl": int(bad_hl.sum())}))
        df.loc[bad_h, "h"] = max_oc[bad_h]
        df.loc[bad_l, "l"] = min_oc[bad_l]
        both = bad_hl
        if both.any():
            fix_h = np.fmax.reduce([df.loc[both, "o"].values, df.loc[both, "h"].values, df.loc[both, "c"].values])
            fix_l = np.fmin.reduce([df.loc[both, "o"].values, df.loc[both, "l"].values, df.loc[both, "c"].values])
            df.loc[both, "h"] = fix_h
            df.loc[both, "l"] = fix_l
        _set_flag(dq_flags, bad_h | bad_l | bad_hl, "OHLC_INVARIANT")

    # отрицательные объёмы
    if cfg.forbid_negative_volume:
        neg_v = df["v"] < 0
        if neg_v.any():
            cnt = int(neg_v.sum())
            issues.append(Issue(ts=df.index.max(), code="NEG_VOLUME", severity="error",
                                msg=f"Отрицательный объём: {cnt}, заменён на 0.0",
                                symbol=symbol, tf=tf, extra={"count": cnt}))
            df.loc[neg_v, "v"] = 0.0
            _set_flag(dq_flags, neg_v, "NEG_VOLUME")
    return df


def _fill_missing_if_small(df: pd.DataFrame, issues: List[Issue], symbol: Optional[str], tf: str, cfg: QualityConfig,
                           dq_flags: pd.Series) -> pd.DataFrame:
    if tf != "1m" or df.empty:
        return df
    full = pd.date_range(df.index.min(), df.index.max(), freq=_tf_freq(tf), tz="UTC")
    miss = full.difference(df.index)
    if not len(miss):
        return df
    missing_rate = 1.0 - (len(df.index) / len(full)) if len(full) else 0.0
    issues.append(Issue(ts=df.index.max(), code="MISSING_BARS", severity="warning",
                        msg=f"Пропуски: {len(miss)} ({missing_rate:.6%})", symbol=symbol, tf=tf,
                        extra={"missing": int(len(miss)), "rate": float(missing_rate)}))
    if missing_rate <= cfg.missing_fill_threshold:
        base_close = df["c"].reindex(full).ffill().bfill()
        filled = pd.DataFrame(index=full, columns=df.columns, dtype=float)
        filled.loc[df.index, ["o", "h", "l", "c", "v"]] = df[["o", "h", "l", "c", "v"]].values
        if "t" in df.columns:
            filled["t"] = 0.0
            filled.loc[df.index, "t"] = df["t"].values
        # синтетика
        filled.loc[miss, "c"] = base_close.loc[miss]
        filled.loc[miss, "o"] = base_close.loc[miss]
        filled.loc[miss, "h"] = base_close.loc[miss]
        filled.loc[miss, "l"] = base_close.loc[miss]
        filled.loc[miss, "v"] = 0.0
        if "is_gap" not in filled.columns:
            filled["is_gap"] = False
        filled.loc[miss, "is_gap"] = True
        issues.append(Issue(ts=df.index.max(), code="MISSING_FILLED", severity="info",
                            msg=f"Заполнено синтетикой: {len(miss)} минут", symbol=symbol, tf=tf))
        # выставляем флаги на вставленные точки
        _set_flag(dq_flags, filled.index.isin(miss), "MISSING_BARS")
        _set_flag(dq_flags, filled.index.isin(miss), "MISSING_FILLED")
        return filled.sort_index()
    return df


def _spike_scan(df: pd.DataFrame, issues: List[Issue], symbol: Optional[str], tf: str, cfg: QualityConfig,
                dq_flags: pd.Series) -> None:
    if df.shape[0] < cfg.spike_window + 5:
        return
    c = df["c"].astype(float)
    c = c.replace(0.0, np.nan).ffill()
    r = np.log(c).diff()
    med = r.rolling(cfg.spike_window, min_periods=cfg.spike_window // 2).median()
    mad = (r - med).abs().rolling(cfg.spike_window, min_periods=cfg.spike_window // 2).median()
    thresh = cfg.spike_k * (mad + 1e-12)
    spikes = (r - med).abs() > thresh
    if spikes.any():
        idx = df.index[spikes.fillna(False)]
        for ts in idx:
            issues.append(Issue(ts=ts, code="PRICE_SPIKE", severity="warning",
                                msg="Аномальный ценовой скачок по MAD", symbol=symbol, tf=tf))
        _set_flag(dq_flags, spikes.fillna(False).values, "PRICE_SPIKE")


def _flat_scan(df: pd.DataFrame, issues: List[Issue], symbol: Optional[str], tf: str, cfg: QualityConfig,
               dq_flags: pd.Series) -> None:
    v0 = (df["v"] == 0.0).astype(int)
    runs = (v0.diff(1) != 0).cumsum()
    lengths = v0.groupby(runs).transform("sum")
    long_zero = (v0.eq(1) & (lengths >= cfg.flat_streak_threshold))
    if long_zero.any():
        first_ts = df.index[long_zero.idxmax()]
        issues.append(Issue(ts=first_ts, code="LONG_ZERO_VOLUME", severity="info",
                            msg=f"Длинная серия нулевого объёма ≥{cfg.flat_streak_threshold}", symbol=symbol, tf=tf))
        _set_flag(dq_flags, long_zero.values, "LONG_ZERO_VOLUME")


# ---- Публичный интерфейс -----------------------------------------------------

def validate(df: pd.DataFrame,
             *,
             tf: str = "1m",
             symbol: Optional[str] = None,
             repair: bool = True,
             config: Optional[QualityConfig] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Основной вход C2. Возвращает детерминированно очищенный df и журнал issues.
    Вход:
      df — DataFrame со столбцами o,h,l,c,v (+ опциональные t,is_gap), индекс — tz-aware DatetimeIndex (UTC), правая граница бара.
    Выход:
      df_clean — тот же формат + колонки dq_flags(int32), dq_notes(str)
      issues   — DataFrame журнала проблем (ts, code, severity, msg, ...)
    """
    cfg = config or QualityConfig()
    issues: List[Issue] = []

    if df.empty:
        return df.copy(), issues_to_frame(issues)

    # Индекс и базовые инварианты
    out = _ensure_index(df.copy(), issues, symbol, tf, cfg)

    # Битовые флаги на все строки (инициализация)
    dq_flags = pd.Series(np.zeros(len(out), dtype=np.int32), index=out.index)

    # Типы и NaN
    out = _coerce_types(out, issues, symbol, tf)

    # OHLC-инварианты и отрицательные объёмы
    if repair:
        out = _fix_ohlc_invariants(out, issues, symbol, tf, cfg, dq_flags)

    # Пропуски 1m: допустимое заполнение
    if repair:
        out = _fill_missing_if_small(out, issues, symbol, tf, cfg, dq_flags)

    # Аномалии: спайки и длинные плоские сегменты
    _spike_scan(out, issues, symbol, tf, cfg, dq_flags)
    _flat_scan(out, issues, symbol, tf, cfg, dq_flags)

    # Финальная сортировка и устранение дублей индекса
    out = out[~out.index.duplicated(keep="last")].sort_index()

    # Колонки dq_flags/dq_notes
    out["dq_flags"] = dq_flags.reindex(out.index, fill_value=0).astype("int32")
    # dq_notes: для ненулевых флагов строим список кодов
    inv_map = {v: k for k, v in DQ_BITS.items()}
    def _flags_to_notes(val: int) -> str:
        if val == 0:
            return ""
        bits = []
        x = val
        b = 0
        while x:
            if x & 1:
                bits.append(inv_map.get(b, str(b)))
            x >>= 1
            b += 1
        return ";".join(bits)
    out["dq_notes"] = out["dq_flags"].apply(_flags_to_notes)

    return out, issues_to_frame(issues)
