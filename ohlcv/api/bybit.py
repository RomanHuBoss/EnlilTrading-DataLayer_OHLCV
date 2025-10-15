# Клиент загрузки OHLCV с Bybit v5 Market API.
# Документация: https://bybit-exchange.github.io/docs/v5/market/kline
import time
import math
import hashlib
import hmac
from typing import Iterable, List, Optional, Dict
from datetime import datetime, timezone
import requests

BASE_URL = "https://api.bybit.com"

def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def _sign(api_secret: str, payload: str) -> str:
    return hmac.new(api_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

def fetch_klines_1m(
    symbol: str,
    start: datetime,
    end: datetime,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    limit: int = 1000,
    sleep_sec: float = 0.2,
) -> List[Dict]:
    """
    Загрузка 1m OHLCV с публичного Market Kline.
    Возвращает список баров (dict) в порядке возрастания по времени.

    Поля ответа Bybit v5 (spot):
      - start: ms
      - open, high, low, close: str
      - volume: str
      - turnover: str
    """
    path = "/v5/market/kline"
    interval = "1"  # 1-minute
    result: List[Dict] = []

    start_ms = _ms(start)
    end_ms = _ms(end)

    cursor = start_ms
    while cursor < end_ms:
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "start": cursor,
            "end": min(end_ms, cursor + limit * 60_000),  # приблизительно limit баров
            "limit": str(limit),
        }
        headers = {}
        # Маркет-эндпоинт не требует подписи. Оставлено для совместимости при необходимости.
        if api_key and api_secret:
            headers["X-BAPI-API-KEY"] = api_key

        r = requests.get(BASE_URL + path, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit error: {data.get('retCode')} {data.get('retMsg')}")
        rows = data["result"]["list"]
        if not rows:
            # Сдвиг курсора на limit минут вперёд, чтобы не зациклиться
            cursor += limit * 60_000
            time.sleep(sleep_sec)
            continue

        # Ответ приходит в порядке от нового к старому — разворачиваем
        rows = list(reversed(rows))
        for row in rows:
            ts_ms = int(row[0])
            if ts_ms < start_ms or ts_ms >= end_ms:
                continue
            result.append({
                "ts": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(),
                "o": float(row[1]),
                "h": float(row[2]),
                "l": float(row[3]),
                "c": float(row[4]),
                "v": float(row[5]),
                "t": float(row[6]) if len(row) > 6 and row[6] is not None else None,
            })
        cursor = int(rows[-1][0]) + 60_000
        time.sleep(sleep_sec)
    return result
