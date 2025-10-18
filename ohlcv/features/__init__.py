"""
C3.Features.Core — публичный API пакета.
Экспортирует compute_features, DEFAULTS, normalize_and_validate, BuildMeta.
Версия пакета — __version__.
"""

from .core import compute_features, DEFAULTS
from .schema import normalize_and_validate
from .utils import BuildMeta

__all__ = [
    "compute_features",
    "DEFAULTS",
    "normalize_and_validate",
    "BuildMeta",
    "__version__",
]

# Версия компонента C3. Поддерживается CLI и BuildMeta.
__version__ = "0.1.0"
