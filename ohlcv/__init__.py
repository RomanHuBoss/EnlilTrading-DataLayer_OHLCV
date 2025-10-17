from __future__ import annotations

# Версия пакета — максимально надёжное получение.
try:
    # Локальный модуль, если присутствует (генерируемый при сборке).
    from .version import __version__  # type: ignore[attr-defined]
except Exception:
    try:
        # Установленное имя дистрибутива из pyproject.toml
        from importlib.metadata import PackageNotFoundError, version

        try:
            __version__ = version("ohlcv-pipeline")  # type: ignore[assignment]
        except PackageNotFoundError:
            __version__ = "0.0.0"  # type: ignore[assignment]
    except Exception:  # крайний fallback
        __version__ = "0.0.0"  # type: ignore[assignment]

__all__ = ["__version__"]
