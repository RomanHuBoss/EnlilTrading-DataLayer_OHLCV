from __future__ import annotations

import numpy as np

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore

    _HMM_AVAILABLE = True
except Exception:  # noqa: BLE001
    _HMM_AVAILABLE = False


def fit_predict_hmm_rv(rv_daily: np.ndarray, n_states: int = 2) -> np.ndarray:
    """Возвращает вероятности состояний calm по rv_daily.
    Если hmmlearn недоступен — возбуждает RuntimeError для fallback на квантильный метод.
    """
    if not _HMM_AVAILABLE:
        raise RuntimeError("hmmlearn not available")
    x = rv_daily.reshape(-1, 1).astype(float)
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100, random_state=0)
    model.fit(x)
    post = model.predict_proba(x)
    # Считаем, что компонент с меньшей дисперсией = calm
    variances = model.covars_.reshape(-1)
    calm_idx = int(np.argmin(variances))
    return post[:, calm_idx]
