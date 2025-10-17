from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version


def _get_version() -> str:
    # 1) локальная сборка может положить ohlcv/version.py c __version__
    try:
        from .version import __version__ as v  # type: ignore[no-redef]

        return str(v)
    except Exception:
        pass
    # 2) установленный дистрибутив
    try:
        return _pkg_version("ohlcv-pipeline")
    except PackageNotFoundError:
        return "0.0.0"


__version__ = _get_version()
__all__ = ["__version__"]
