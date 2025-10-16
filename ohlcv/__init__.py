# Пакет и публичные экспортируемые символы.

# Пакет и публичные экспортируемые символы.

from .version import __version__

# Удобные реэкспорты C2 (необязательны, но упрощают импорт в скриптах/тетрадках):
#   from ohlcv import quality_validate, QualityConfig, DQ_BITS
try:
    from .quality import validate as quality_validate, QualityConfig, DQ_BITS  # noqa: F401
except Exception:
    # На случай частичных установок без подпакета quality.
    quality_validate = None  # type: ignore
    QualityConfig = None  # type: ignore
    DQ_BITS = None  # type: ignore

__all__ = [
    "__version__",
    "quality_validate",
    "QualityConfig",
    "DQ_BITS",
]
