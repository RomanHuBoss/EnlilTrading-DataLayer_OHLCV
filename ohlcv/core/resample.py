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
            df_1m = df_1m.asfreq("min")
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
    # closed='left' устраняет лишний пустой бин на границе 00:00, оставляя две 5-минутки из 10 минут данных
    agg = df_1m.resample(rule, label="right", closed="left").agg(ohlc_dict).dropna()
    return agg[["o","h","l","c","v"]]
