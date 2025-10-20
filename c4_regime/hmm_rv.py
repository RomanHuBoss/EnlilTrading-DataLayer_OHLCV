# c4_regime/hmm_rv.py
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["fit_predict_hmm_rv"]


def _prep_series(rv: np.ndarray) -> np.ndarray:
    """Подготовить дневную RV: числовой вектор без NaN/Inf, мягкое сглаживание.
    Возвращает массив shape (n, 1) для подачи в HMM.
    """
    x = np.asarray(rv, dtype=float).reshape(-1)
    if x.size == 0:
        return x.reshape(-1, 1)

    # Канонизация значений
    x = np.where(np.isfinite(x), x, np.nan)
    s = pd.Series(x)
    if s.isna().any():
        s = s.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    # Робаст‑сглаживание: медианный фильтр на коротком окне
    if len(s) >= 5:
        s = s.rolling(window=5, min_periods=1, center=True).median()

    # Переход в лог‑шкалу
    y = np.log1p(np.maximum(s.to_numpy(), 0.0))
    return y.reshape(-1, 1)


def fit_predict_hmm_rv(rv_daily: np.ndarray, n_states: int = 2, *, random_state: int = 42) -> np.ndarray:
    """Вернуть p_calm ∈ [0,1] длины n по HMM над лог(RV+1).

    Требует ``hmmlearn``. При недоступности/нехватке данных бросает исключение
    — вызывающий код обязан перехватить и применить фолбэк (см. d4_hmm_rv_or_quantile).
    """
    try:
        from hmmlearn.hmm import GaussianHMM  # type: ignore
    except Exception as e:  # noqa: PERF203
        raise ImportError("hmmlearn не установлен") from e

    X = _prep_series(rv_daily)
    n = int(X.shape[0])
    if n < max(10, int(n_states) * 3):
        raise ValueError("слишком мало наблюдений для HMM")

    # Инициализация и обучение
    n_states = int(n_states)
    # Инициируем средние точками 10..90 перцентили, ковариации — общая вариация
    mu_low = float(np.nanpercentile(X, 10))
    mu_high = float(np.nanpercentile(X, 90))
    init_means = np.linspace(mu_low, mu_high, n_states, dtype=float).reshape(-1, 1)
    init_cov = np.full((n_states, 1), float(np.nanvar(X) + 1e-6))

    # init_params='st': стартовые вероятности и матрица переходов инициализируются внутри fit,
    # а наши means_/covars_ используются как стартовые и НЕ переинициализируются.
    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=100,
        tol=1e-3,
        random_state=int(random_state),
        verbose=False,
        init_params="st",
        params="stmc",
    )

    hmm.means_ = init_means
    hmm.covars_ = init_cov

    try:
        hmm.fit(X)
    except Exception as e:
        raise RuntimeError("HMM не сошлась") from e

    try:
        _, post = hmm.score_samples(X)
    except Exception as e:
        raise RuntimeError("не удалось получить постериоры HMM") from e

    means = hmm.means_.reshape(-1)
    calm_idx = int(np.nanargmin(means))

    p_calm = np.asarray(post[:, calm_idx], dtype=float)
    return np.clip(p_calm, 0.0, 1.0)
