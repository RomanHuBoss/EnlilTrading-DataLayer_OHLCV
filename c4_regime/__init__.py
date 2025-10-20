# c4_regime/__init__.py
from __future__ import annotations

# Версия берётся из пакета ohlcv при наличии
try:  # pragma: no cover
    from ohlcv.version import __version__ as _VER  # type: ignore
except Exception:  # pragma: no cover
    _VER = "0.0.0"
__version__ = _VER

# Базовое API C4
from .core import RegimeConfig, infer_regime
from .detectors import (
    d1_adx_trend,
    d2_donch_width,
    d3_bocpd_proxy,
    d4_hmm_rv_or_quantile,
)
from .bocpd import run_length_bocpd
from .hmm_rv import fit_predict_hmm_rv
from .errors import (
    ErrorCodes,
    BaseC4Error,
    ConfigError,
    DataSchemaError,
    MissingFeatureError,
    LookbackError,
    RuntimeIssueError,
)

__all__ = [
    "__version__",
    # core
    "RegimeConfig",
    "infer_regime",
    # detectors
    "d1_adx_trend",
    "d2_donch_width",
    "d3_bocpd_proxy",
    "d4_hmm_rv_or_quantile",
    # models/helpers
    "run_length_bocpd",
    "fit_predict_hmm_rv",
    # errors
    "ErrorCodes",
    "BaseC4Error",
    "ConfigError",
    "DataSchemaError",
    "MissingFeatureError",
    "LookbackError",
    "RuntimeIssueError",
]
