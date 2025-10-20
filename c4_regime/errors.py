# c4_regime/errors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

__all__ = [
    "ErrorCodes",
    "BaseC4Error",
    "ConfigError",
    "DataSchemaError",
    "MissingFeatureError",
    "LookbackError",
    "RuntimeIssueError",
]


class ErrorCodes:
    """Стабильные строковые коды ошибок (для логов/метрик/тестов)."""

    E_MISSING_FEATURE = "E_MISSING_FEATURE"  # отсутствует требуемая колонка/признак
    E_LOOKBACK = "E_LOOKBACK"  # недостаточная длина истории для расчёта
    E_CFG = "E_CFG"  # ошибки конфигурации/валидации аргументов
    E_SCHEMA = "E_SCHEMA"  # нарушения C1/C2/C3 схем
    E_RUNTIME = "E_RUNTIME"  # прочие ошибки исполнения


@dataclass
class BaseC4Error(Exception):
    message: str
    code: str = ErrorCodes.E_RUNTIME
    ctx: Optional[Dict[str, object]] = None

    def __str__(self) -> str:  # лаконичное сообщение + код
        if self.ctx:
            return f"{self.code}: {self.message} | ctx={self.ctx}"
        return f"{self.code}: {self.message}"


class ConfigError(BaseC4Error):
    def __init__(self, message: str, **ctx):
        super().__init__(message, code=ErrorCodes.E_CFG, ctx=ctx or None)


class DataSchemaError(BaseC4Error):
    def __init__(self, message: str, **ctx):
        super().__init__(message, code=ErrorCodes.E_SCHEMA, ctx=ctx or None)


class MissingFeatureError(BaseC4Error, KeyError):
    def __init__(self, feature: str, **ctx):
        msg = f"missing feature '{feature}'"
        ctx2 = {"feature": feature}
        ctx2.update(ctx)
        super().__init__(msg, code=ErrorCodes.E_MISSING_FEATURE, ctx=ctx2)


class LookbackError(BaseC4Error, IndexError):
    def __init__(self, need: int, have: int, **ctx):
        msg = f"lookback not satisfied: need>={need}, have={have}"
        ctx2 = {"need": int(need), "have": int(have)}
        ctx2.update(ctx)
        super().__init__(msg, code=ErrorCodes.E_LOOKBACK, ctx=ctx2)


class RuntimeIssueError(BaseC4Error, RuntimeError):
    def __init__(self, message: str, **ctx):
        super().__init__(message, code=ErrorCodes.E_RUNTIME, ctx=ctx or None)
