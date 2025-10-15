# Клиент загрузки OHLCV с Bybit v5 Market API c поддержкой потоковой выдачи чанков.
# Документация: https://bybit-exchange.github.io/docs/v5/market/kline
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


def _http_get(path: str, params: Dict, headers: Dict, timeout: int, max_retries: int, backoff_base: float) -> Response:
    """
    Надёжный GET с экспоненциальным бэкоффом.
    Исключения/коды ≠200 → повтор до max_retries.
    """
    url = BASE_URL + path
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            # Бывает 429/5xx — уйдём на ретрай
            if r.status_code == 200:
                return r
            # Жёсткие ошибки — тоже ретрай
        except requests.RequestException:
            pass
        # Бэкофф: base * 2^attempt
        time.sleep(backoff_base * (2 ** attempt))
    # Последняя попытка без сна — пусть бросит исключение вызывающему
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r


def iter_klines_1m(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    category: str = "spot",             # допустимо: spot|linear|inverse
    api_key: Optional[str] = None,      # read-only ключи по постановке
    api_secret: Optional[str] = None,   # не используется на публичном эндпоинте
    limit: int = 1000,                  # макс. на Bybit v5
    sleep_sec: float = 0.2,             # пауза между запросами
    max_retries: int = 5,               # надёжность на сетевых/5xx/429
    timeout: int = 30                   # таймаут HTTP, сек
) -> Iterable[List[Dict]]:
    """
    Итеративная загрузка 1m OHLCV: выдаёт чанки по мере получения.
    Каждый чанк — список словарей баров, отсортированный по возрастанию времени.
    Гарантии:
      - временное окно [start, end) в UTC
      - дубликаты на границах отфильтрованы вызывающим слоем по индексу

    Формат бара:
      {"ts": ISO8601 UTC, "o": float, "h": float, "l": float, "c": float, "v": float, "t": float|None}
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
            # ограничиваем окно так, чтобы сервер вернул не более ~limit баров
            "start": cursor,
            "end": min(end_ms, cursor + limit * 60_000),
            "limit": str(limit),
        }
        headers: Dict[str, str] = {}
        if api_key:
            # Публичный маркет-эндпоинт не требует подписи, но ключ допустим.
            headers["X-BAPI-API-KEY"] = api_key

        r = _http_get(path, params, headers, timeout=timeout, max_retries=max_retries, backoff_base=sleep_sec)
        # Если выше не бросило — статус 200 гарантирован
        data = r.json()
        if data.get("retCode") != 0:
            # Прерывание — пусть внешний слой решает, ретраить ли весь цикл
            raise RuntimeError(f"Bybit error: {data.get('retCode')} {data.get('retMsg')}")

        rows = data.get("result", {}).get("list") or []
        if not rows:
            # Пусто: сдвигаем курсор на limit минут, чтобы не зациклиться при редких дырах
            cursor += limit * 60_000
            time.sleep(sleep_sec)
            continue

        # Bybit возвращает от нового к старому — разворачиваем.
        rows = list(reversed(rows))

        chunk: List[Dict] = []
        for row in rows:
            # Формат v5: [start, open, high, low, close, volume, turnover, ...]
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
            # Следующая страница — от последней минуты + 60_000 мс
            cursor = int(rows[-1][0]) + 60_000
        else:
            # Если всё отфильтровали границами окна — сдвиг на limit минут
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
    """
    Совместимая обёртка поверх iter_klines_1m: возвращает единый список баров.
    Используется там, где нужен не поток, а полный буфер.
    """
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
