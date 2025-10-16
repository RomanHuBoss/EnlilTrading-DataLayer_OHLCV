# ohlcv/core/resample.py
# Ресемплинг OHLCV с 1m на агрегированные ТФ (5m/15m/1h).
# Требования:
# - Индекс входа: tz-aware UTC DatetimeIndex (минутные правые границы).
# - Столбцы: o,h,l,c,v; опционально t (turnover) и is_gap (флаг синтетики календаризации).
# - Политика окна: label="right", closed="left" — правая граница агрегата совпадает с правой границей окна.
# - Агрегации: o=first, h=max, l=min, c=last, v=sum, t=sum, is_gap=max (если есть).
# - Выход: индекс — правая граница целевого ТФ; порядок столбцов: o,h,l,c,v,(t?),(is_gap?).
from typing import Dict, Literal

import pandas as pd

from ..utils.timeframes import tf_to_pandas_rule


def resample_ohlcv(df_1m: pd.DataFrame, to_tf: Literal["5m", "15m", "1h"]) -> pd.DataFrame:
    """
    Вход:
        df_1m: DataFrame с индексом DatetimeIndex(UTC) и колонками ['o','h','l','c','v'] (+ опционально 't','is_gap').
        to_tf: '5m' | '15m' | '1h'
    Выход:
        DataFrame агрегированного таймфрейма с теми же колонками (+ опциональными), индекс — правая граница бара.
    """
    if not isinstance(df_1m.index, pd.DatetimeIndex) or df_1m.index.tz is None:
        raise ValueError("Ожидается tz-aware DatetimeIndex (UTC)")
    if not df_1m.index.is_monotonic_increasing:
        df_1m = df_1m.sort_index()

    # Пытаемся выставить частоту для более корректного resample (не критично).
    if df_1m.index.freq is None:
        try:
            df_1m = df_1m.asfreq("min")
        except Exception:
            pass

    rule = tf_to_pandas_rule(to_tf)

    agg: Dict[str, str] = {
        "o": "first",
        "h": "max",
        "l": "min",
        "c": "last",
        "v": "sum",
    }
    if "t" in df_1m.columns:
        agg["t"] = "sum"
    if "is_gap" in df_1m.columns:
        # Прокидываем флаг «есть синтетика внутри окна»
        # Для bool/max pandas корректно агрегирует: True, если в окне есть хотя бы один True.
        agg["is_gap"] = "max"

    out = df_1m.resample(rule, label="right", closed="left").agg(agg)

    # Удаляем полностью пустые агрегаты (на случай пропусков входа).
    out = out.dropna(how="all")

    # Базовый порядок столбцов + опциональные (если были на входе).
    cols = ["o", "h", "l", "c", "v"]
    if "t" in out.columns:
        cols.append("t")
    if "is_gap" in out.columns:
        cols.append("is_gap")

    return out[cols]
