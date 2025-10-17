"""C4.Regime — онлайн-детекция рыночного режима (trend/flat + high-RV).
Версия компонента следует семверу.
"""
from __future__ import annotations

__all__ = [
    "infer_regime",
]

__version__ = "0.1.0"

from .core import infer_regime  # noqa: E402
