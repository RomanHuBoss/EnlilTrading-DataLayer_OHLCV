from __future__ import annotations

import numpy as np
import pandas as pd

# Упрощённая BOCPD: оценка длины текущего "пробега" (run_length) на основе
# последовательности x и параметра lambda. О(t) Online, но реализовано в векторном виде.


def sigmoid(x: np.ndarray, k: float = 1.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * x))


def run_length_bocpd(
    x: np.ndarray,
    lam: float = 200.0,
    L_min: float = 200.0,
    sigma_L: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает (run_length, p_trend, trend_flag, flat_flag).
    Не строгая реализация Adams & MacKay, но стабильная и детерминированная.
    Идея: вероятность смены ~ 1/lam, дополнительный триггер — |z| больших амплитуд.
    """
    n = len(x)
    run = np.zeros(n, dtype=np.int32)
    # Скользящая оценка σ для z-скор
    x_s = pd.Series(x)
    # Небольшое окно для локальной вариативности, не менее 50
    W = max(50, int(min(lam, 300)))
    mu = x_s.rolling(W, min_periods=1).mean().to_numpy()
    sd = x_s.rolling(W, min_periods=1).std(ddof=1).fillna(0.0).replace(0.0, 1e-12).to_numpy()
    z = (x - mu) / sd

    # Вероятность hazard
    p_hazard = 1.0 / max(lam, 1.0)

    rng = np.random.default_rng(0)  # детерминированность
    # Вместо случайного сэмплинга используем детерминированную эвристику:
    # считаем, что смена происходит, когда p_hazard > 0.0 и |z| превышает 2.0

    for i in range(n):
        if i == 0:
            run[i] = 0
            continue
        run[i] = run[i - 1] + 1
        if (abs(z[i]) >= 2.0) or (run[i] > 4 * lam):
            # Смена режима
            run[i] = 0

    # Оценка тренд-скора: большие пробеги → trend
    p_trend = sigmoid((run - L_min) / max(sigma_L, 1.0))
    trend_flag = (p_trend >= 0.5).astype(np.int8)
    flat_flag = (1 - trend_flag).astype(np.int8)
    return run, p_trend, trend_flag, flat_flag
