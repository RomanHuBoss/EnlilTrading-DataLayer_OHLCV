# ohlcv/api/bybit.py — Market v5: 1m OHLCV (чанки/буфер) + instruments-info.launchTime
# Документация:
#   Kline:            https://bybit-exchange.github.io/docs/v5/market/kline
#   InstrumentsInfo:  https://bybit-exchange.github.io/docs/v5/market/instrument
import time
from typing import Iterable, List, Optional, Dict
from datetime import datetime, timezone
import hashlib
import hmac
import requests
from requests import Response

BASE_URL = "https://api.bybit.com"


def _ms(dt: datetime) -> int:
    """Datetime → миллисекунды UNIX. Ожидается tz-aware UTC."""
    return int(dt.timestamp() * 1000)


def _sign(api_secret: str, payload: str) -> str:
    """HMAC-SHA256 подпись. Для публичных маркет-эндпоинтов не требуется."""
    return hmac.new(api_secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _http_get(path: str, params: Dict, headers: Dict, *, timeout: int, max_retries: int, backoff_base: float) -> Response:
    """
    Надёжный GET с экспоненциальным бэкоффом.
    Исключения/коды ≠200 → повтор до max_retries.
    """
    url = BASE_URL + path
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(backoff_base * (2 ** attempt))
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r


def get_launch_time(symbol: str, *, category: str = "linear", timeout: int = 30, max_retries: int = 5) -> Optional[datetime]:
    """
    Возвращает UTC datetime запуска инструмента для категорий linear|inverse (поле launchTime в мс).
    Для spot, как правило, поле отсутствует → None.
    """
    path = "/v5/market/instruments-info"
    params = {"category": category, "symbol": symbol}
    r = _http_get(path, params, headers={}, timeout=timeout, max_retries=max_retries, backoff_base=0.2)
    data = r.json()
    if data.get("retCode") != 0:
        return None
    items = (data.get("result") or {}).get("list") or []
    if not items:
        return None
    item = items[0]
    lt = item.get("launchTime")
    if lt is None:
        return None
    try:
        ts_ms = int(lt)
        return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).replace(second=0, microsecond=0)
    except Exception:
        return None


def iter_klines_1m(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    category: str = "spot",             # spot|linear|inverse
    api_key: Optional[str] = None,      # read-only
    api_secret: Optional[str] = None,   # не требуется для публичного маркет-эндпоинта
    limit: int = 1000,
    sleep_sec: float = 0.2,
    max_retries: int = 5,
    timeout: int = 30
) -> Iterable[List[Dict]]:
    """
    Итеративная загрузка 1m OHLCV: выдаёт чанки по мере получения.
    Формат: {"ts","o","h","l","c","v","t?"}; окно [start, end), UTC.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start/end должны быть tz-aware UTC")
    if end <= start:
        return

    path = "/v5/market/kline"
    interval = "1"

    start_ms = _ms(start)
    end_ms = _ms(end)

    cursor = start_ms
    while cursor < end_ms:
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "start": cursor,
            "end": min(end_ms, cursor + limit * 60_000),
            "limit": str(limit),
        }
        headers: Dict[str, str] = {}
        if api_key:
            headers["X-BAPI-API-KEY"] = api_key

        r = _http_get(path, params, headers, timeout=timeout, max_retries=max_retries, backoff_base=sleep_sec)
        data = r.json()
        if data.get("retCode") != 0:
            raise RuntimeError(f"Bybit error: {data.get('retCode')} {data.get('retMsg')}")

        rows = (data.get("result") or {}).get("list") or []
        if not rows:
            cursor += limit * 60_000
            time.sleep(sleep_sec)
            continue

        # Ответ идёт от нового к старому — разворот.
        rows = list(reversed(rows))

        chunk: List[Dict] = []
        for row in rows:
            # row: [start, open, high, low, close, volume, turnover, ...]
            ts_ms = int(row[0])
            if ts_ms < start_ms or ts_ms >= end_ms:
                continue
            chunk.append({
                "ts": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(),
                "o": float(row[1]),
                "h": float(row[2]),
                "l": float(row[3]),
                "c": float(row[4]),
                "v": float(row[5]),
                "t": float(row[6]) if len(row) > 6 and row[6] is not None else None,
            })

        if chunk:
            yield chunk
            cursor = int(rows[-1][0]) + 60_000
        else:
            cursor += limit * 60_000

        time.sleep(sleep_sec)


def fetch_klines_1m(
    symbol: str,
    start: datetime,
    end: datetime,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    limit: int = 1000,
    sleep_sec: float = 0.2,
    *,
    category: str = "spot",
    max_retries: int = 5,
    timeout: int = 30
) -> List[Dict]:
    """Буферизующая обёртка над iter_klines_1m."""
    out: List[Dict] = []
    for chunk in iter_klines_1m(
        symbol,
        start,
        end,
        category=category,
        api_key=api_key,
        api_secret=api_secret,
        limit=limit,
        sleep_sec=sleep_sec,
        max_retries=max_retries,
        timeout=timeout,
    ):
        out.extend(chunk)
    return out
