# Ресемплинг OHLCV с 1m на агрегированные ТФ.
import pandas as pd
from typing import Literal
from ..utils.timeframes import tf_to_pandas_rule

def resample_ohlcv(df_1m: pd.DataFrame, to_tf: Literal["5m","15m","1h"]) -> pd.DataFrame:
    """
    Вход: DataFrame со столбцами ['o','h','l','c','v'] и DatetimeIndex (UTC, минутные границы).
    Выход: DataFrame в целевом ТФ с теми же столбцами; индекс — правая граница бара.
    """
    if not isinstance(df_1m.index, pd.DatetimeIndex) or df_1m.index.tz is None:
        raise ValueError("Ожидается tz-aware DatetimeIndex (UTC)")
    if df_1m.index.freq is None:
        try:
            df_1m = df_1m.asfreq("T")
        except Exception:
            pass

    rule = tf_to_pandas_rule(to_tf)

    ohlc_dict = {
        "o": "first",
        "h": "max",
        "l": "min",
        "c": "last",
        "v": "sum",
    }
    agg = df_1m.resample(rule, label="right", closed="right").agg(ohlc_dict).dropna()
    agg = agg[["o","h","l","c","v"]]
    return agg
