# c4_regime/bocpd.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

__all__ = ["run_length_bocpd"]


def _rolling_quantile_abs(x: np.ndarray, window: int, q: float) -> np.ndarray:
    """Квантиль по |x| на скользящем окне размера *window*.
    min_periods берётся как max(5, window//5). Первые значения bfill→ffill.
    """
    s = pd.Series(np.asarray(x, dtype=float))
    w = int(max(5, window))
    mp = max(5, w // 5)
    qv = s.abs().rolling(w, min_periods=mp).quantile(float(q))
    qv = qv.bfill().fillna(method="ffill").fillna(float(np.nanmedian(np.abs(x))))
    return qv.to_numpy()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def run_length_bocpd(
    x: np.ndarray,
    lam: float = 200.0,
    L_min: float = 80.0,
    sigma_L: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Прокси‑алгоритм BOCPD для потока приращений x.
    Возвращает кортеж массивов длиной len(x):
      run_length, p_trend∈[0,1], trend_flag∈{0,1}, flat_flag∈{0,1}.

    Логика:
      — Порог редких изменений: q = quantile(|x|, 0.80) на окне *lam* (min_periods≈lam/5).
      — Если |x_t| < q_t → продолжаем пробег, иначе разрыв (run=0).
      — Вероятность тренда: σ((run − L_min)/sigma_L).
    Совместимо с c4_regime.detectors.d3_bocpd_proxy.
    """
    x = np.asarray(x, dtype=float)
    n = int(x.shape[0])
    if n == 0:
        empty = np.asarray([], dtype=float)
        empty_i8 = np.asarray([], dtype=np.int8)
        return empty, empty, empty_i8, empty_i8

    lam = float(lam)
    L_min = float(L_min)
    sigma_L = float(max(sigma_L, 1.0))

    # Квантильный порог по окну lam
    thr = _rolling_quantile_abs(x, window=int(max(20, lam)), q=0.80)

    below = np.less(np.abs(x), thr).astype(np.int8)

    run = np.zeros(n, dtype=float)
    for i in range(n):
        if i == 0:
            run[i] = 0.0
        else:
            run[i] = (run[i - 1] + 1.0) * float(below[i])

    p_trend = _sigmoid((run - L_min) / sigma_L).astype(float)
    trend_flag = (p_trend >= 0.5).astype(np.int8)
    flat_flag = (1 - trend_flag).astype(np.int8)

    return run, p_trend, trend_flag, flat_flag
