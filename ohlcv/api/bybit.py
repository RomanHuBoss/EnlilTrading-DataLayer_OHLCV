# ohlcv/api/bybit.py — Bybit v5 REST, публичные эндпоинты для OHLCV 1m и launchTime
# Все времена — UTC. Возвращаемые бары — правая граница (ts = start_ms + 60_000 - 0?)
# Для согласованности с пайплайном C1/C2 используем ts как правая граница минуты.

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, Generator, Iterable, List, Optional

import requests

BASE_URL = "https://api.bybit.com"
USER_AGENT = "EnlilTrading-DataLayer/1.0 (python-requests)"

# Ограничения Bybit: limit<=1000, интервал 1m передаётся как "1" (минуты)
KLINE_LIMIT = 1000


def _http_get(path: str,
              params: Dict[str, str | int | float | None],
              headers: Optional[Dict[str, str]] = None,
              *,
              timeout: int = 20,
              max_retries: int = 5,
              backoff_base: float = 0.8) -> requests.Response:
    url = f"{BASE_URL}{path}"
    h = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, params={k: v for k, v in params.items() if v is not None}, headers=h, timeout=timeout)
            if r.status_code >= 500:
                # серверные ошибки — ретрай с бэкоффом
                raise requests.HTTPError(f"{r.status_code} {r.text[:200]}")
            return r
        except Exception as e:
            last_err = e
            if attempt >= max_retries:
                break
            sleep = backoff_base * (2 ** attempt)
            time.sleep(sleep)
    # если дошли сюда — ретраи исчерпаны
    if isinstance(last_err, Exception):
        raise last_err
    raise RuntimeError("HTTP GET failed with unknown error")


def _iso_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _rows_from_kline_list(lst: List[List[str]]) -> List[Dict[str, float]]:
    # Формат Bybit v5: [ startMs, open, high, low, close, volume, turnover ] — строки
    out: List[Dict[str, float]] = []
    for it in lst:
        start_ms = int(it[0])
        o = float(it[1]); h = float(it[2]); l = float(it[3]); c = float(it[4])
        v = float(it[5])
        # правую границу минуты — по постановке работаем с правой границей
        ts_iso = _iso_from_ms(start_ms + 60_000)
        row = {"ts": ts_iso, "o": o, "h": h, "l": l, "c": c, "v": v}
        # turnover (опционально)
        try:
            t = float(it[6])
            row["t"] = t
        except Exception:
            pass
        out.append(row)
    return out


def iter_klines_1m(symbol: str,
                   since: datetime,
                   until: datetime,
                   *,
                   api_key: Optional[str] = None,
                   api_secret: Optional[str] = None,
                   category: str = "spot",
                   on_advance: Optional[Callable[[int, int], None]] = None,
                   timeout: int = 20,
                   max_retries: int = 5,
                   sleep_sec: float = 0.2) -> Generator[List[Dict[str, float]], None, None]:
    """
    Генератор чанков 1m баров для диапазона [since, until]. Пагинация вперёд по времени.
    Категории: spot | linear | inverse. Ключи не требуются для публичных методов.
    Возвращает чанки по <=1000 баров в виде списка словарей с ISO ts.
    """
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    if until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)
    if since >= until:
        return

    # У Bybit start/end — миллисекунды. Интервал включительно. interval=1 → 1m
    start_ms = int(since.timestamp() * 1000)
    end_ms = int(until.timestamp() * 1000)

    path = "/v5/market/kline"
    params_base = {
        "category": category,
        "symbol": symbol,
        "interval": "1",
    }

    cursor = start_ms
    while cursor < end_ms:
        # window до end_ms, но Bybit может возвращать меньше limit
        params = dict(params_base)
        params.update({
            "start": cursor,
            "end": end_ms,
            "limit": KLINE_LIMIT,
        })
        r = _http_get(path, params, timeout=timeout, max_retries=max_retries, backoff_base=sleep_sec)
        data = r.json()
        if str(data.get("retCode")) != "0":
            raise RuntimeError(f"Bybit error: {data.get('retCode')} {data.get('retMsg')}")
        result = data.get("result") or {}
        lst = result.get("list") or []
        if not lst:
            # продвигаем курсор, чтобы не зациклиться: шаг 1000 минут или до конца
            cursor = min(cursor + KLINE_LIMIT * 60_000, end_ms)
            if on_advance:
                on_advance(cursor, end_ms)
            time.sleep(sleep_sec)
            continue
        rows = _rows_from_kline_list(sorted(lst, key=lambda x: int(x[0])))
        yield rows
        # продвижение курсора: берём последний start_ms + 60_000
        last_start_ms = int(lst[0][0]) if int(lst[0][0]) > int(lst[-1][0]) else int(lst[-1][0])
        cursor = last_start_ms + KLINE_LIMIT * 0  # не шагать по 1000 минут, а строго к последнему+60_000
        cursor = last_start_ms + 60_000
        if on_advance:
            on_advance(cursor, end_ms)
        time.sleep(sleep_sec)


def fetch_klines_1m(symbol: str,
                    since: datetime,
                    until: datetime,
                    *,
                    api_key: Optional[str] = None,
                    api_secret: Optional[str] = None,
                    category: str = "spot",
                    timeout: int = 20,
                    max_retries: int = 5) -> List[Dict[str, float]]:
    acc: List[Dict[str, float]] = []
    for chunk in iter_klines_1m(symbol, since, until, api_key=api_key, api_secret=api_secret,
                                category=category, timeout=timeout, max_retries=max_retries):
        acc.extend(chunk)
    return acc


def get_launch_time(symbol: str, *, category: str = "spot", timeout: int = 20, max_retries: int = 5) -> Optional[datetime]:
    """
    Возвращает дату/время запуска инструмента (по Bybit v5 instruments-info) для категории.
    Для spot возвращаем минимальную из доступных дат (если есть), иначе None.
    Поле может называться launchTime/createdTime. Возвращаем tz-aware UTC.
    """
    path = "/v5/market/instruments-info"
    params = {
        "category": category,
        "symbol": symbol,
    }
    r = _http_get(path, params, timeout=timeout, max_retries=max_retries)
    data = r.json()
    if str(data.get("retCode")) != "0":
        return None
    result = data.get("result") or {}
    lst = result.get("list") or []
    if not lst:
        return None
    # В ответе могут быть разные контракты/варианты. Берём минимальную доступную дату.
    candidates: List[int] = []
    for it in lst:
        for key in ("launchTime", "createdTime", "listTime"):
            if key in it and it[key] is not None:
                try:
                    candidates.append(int(it[key]))
                except Exception:
                    pass
    if not candidates:
        return None
    ms = min(candidates)
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
