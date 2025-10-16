# Модель и утилиты для журналирования проблем качества данных OHLCV.
# Все времена — UTC. Кодировки проблем стабильные, пригодны для фильтрации в CI.
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import pandas as pd


@dataclass(frozen=True)
class Issue:
    ts: pd.Timestamp            # правая граница бара (UTC)
    code: str                   # стабильный код правила, напр. DUPLICATE, MISALIGNED_TS, NEG_VOLUME, OHLC_INVARIANT
    severity: str               # info | warning | error
    msg: str                    # краткое описание
    symbol: Optional[str] = None
    tf: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Приводим ts к ISO и гарантируем сериализуемость
        d["ts"] = self.ts.isoformat()
        return d


def issues_to_frame(issues: List[Issue]) -> pd.DataFrame:
    if not issues:
        return pd.DataFrame(columns=["ts", "symbol", "tf", "code", "severity", "msg", "extra"])
    rows = [i.to_dict() for i in issues]
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df[["ts", "symbol", "tf", "code", "severity", "msg", "extra"]]
