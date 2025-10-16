# минимальный публичный интерфейс quality-пакета
__all__ = ["QualityConfig", "validate", "DQ_BITS"]

from .validator import QualityConfig, validate, DQ_BITS  # noqa: F401
from . import issues as _issues  # noqa: F401  # импорт для покрытия
