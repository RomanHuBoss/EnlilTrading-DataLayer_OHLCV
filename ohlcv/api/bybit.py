# ohlcv/api/bybit.py — Bybit v5 REST, публичные эндпоинты для OHLCV 1m и launchTime
# Все времена — UTC. Возвращаемые бары — правая граница (ts = start_ms + 60_000).

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Dict, Generator, List, Optional

import requests

BASE_URL = "https://api.bybit.com"
USER_AGENT = "EnlilTrading-DataLayer/1.0 (python-requests)"

# Ограничения Bybit: limit<=1000, интервал 1m передаётся как "1"
KLINE_LIMIT = 1000


def _http_get(
    path: str,
    params: Dict[str, str | int | float | None],
    headers: Optional[Dict[str, str]] = None,
    *,
    timeout: int = 20,
    max_retries: int = 5,
    backoff_base: float = 0.8,
) -> requests.Response:
    url = f"{BASE_URL}{path}"
    h = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(
                url,
                params={k: v for k, v in params.items() if v is not None},
                headers=h,
                timeout=timeout,
            )
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} {r.text[:200]}")
            return r
        except Exception as e:
            last_err = e
            if attempt >= max_retries:
                break
            time.sleep(backoff_base * (2 ** attempt))
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
        ts_iso = _iso_from_ms(start_ms + 60_000)  # правая граница
        row: Dict[str, float] = {"ts": ts_iso, "o": o, "h": h, "l": l, "c": c, "v": v}
        if len(it) > 6:
            try:
                row["t"] = float(it[6])
            except Exception:
                pass
        out.append(row)
    return out


def iter_klines_1m(
    symbol: str,
    since: datetime,
    until: datetime,
    *,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    category: str = "spot",
    on_advance: Optional[Callable[[int, int], None]] = None,
    timeout: int = 20,
    max_retries: int = 5,
    sleep_sec: float = 0.2,
) -> Generator[List[Dict[str, float]], None, None]:
    """
    Генератор чанков 1m баров. Пагинация за счёт сдвига параметра `end` к началу окна.
    API отдаёт список в обратном порядке (по убыванию startTime), забираем по 1000 и двигаем end.
    """
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    if until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)
    if since >= until:
        return

    start_ms = int(since.timestamp() * 1000)
    end_ms_initial = int(until.timestamp() * 1000)

    path = "/v5/market/kline"
    params_base = {
        "category": category,
        "symbol": symbol,
        "interval": "1",
        "limit": KLINE_LIMIT,
    }

    window_end = end_ms_initial
    step_ms = KLINE_LIMIT * 60_000

    while window_end > start_ms:
        params = dict(params_base)
        params["start"] = start_ms
        params["end"] = window_end

        r = _http_get(path, params, timeout=timeout, max_retries=max_retries, backoff_base=sleep_sec)
        data = r.json()
        if str(data.get("retCode")) != "0":
            raise RuntimeError(f"Bybit error: {data.get('retCode')} {data.get('retMsg')}")
        result = data.get("result") or {}
        lst = result.get("list") or []

        if not lst:
            # Пустое окно: сдвиг назад фиксированным шагом
            new_end = max(start_ms, window_end - step_ms)
            if new_end == window_end:
                break
            window_end = new_end
            if on_advance:
                on_advance(window_end, end_ms_initial)
            time.sleep(sleep_sec)
            continue

        # Отсортируем по возрастанию для корректного построения временного ряда
        lst_sorted = sorted(lst, key=lambda x: int(x[0]))
        rows = _rows_from_kline_list(lst_sorted)
        yield rows

        # Новый конец окна — самая ранняя startTime текущего чанка минус 60 секунд
        earliest_start_ms = int(lst_sorted[0][0])
        if earliest_start_ms <= start_ms:
            break
        window_end = earliest_start_ms - 60_000

        if on_advance:
            on_advance(window_end, end_ms_initial)
        time.sleep(sleep_sec)


def fetch_klines_1m(
    symbol: str,
    since: datetime,
    until: datetime,
    *,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    category: str = "spot",
    timeout: int = 20,
    max_retries: int = 5,
) -> List[Dict[str, float]]:
    acc: List[Dict[str, float]] = []
    for chunk in iter_klines_1m(
        symbol,
        since,
        until,
        api_key=api_key,
        api_secret=api_secret,
        category=category,
        timeout=timeout,
        max_retries=max_retries,
    ):
        acc.extend(chunk)
    return acc


def get_launch_time(
    symbol: str,
    *,
    category: str = "spot",
    timeout: int = 20,
    max_retries: int = 5,
) -> Optional[datetime]:
    """
    Дата/время запуска инструмента (v5 instruments-info) для категории.
    Возвращает tz-aware UTC или None.
    """
    path = "/v5/market/instruments-info"
    params = {"category": category, "symbol": symbol}
    r = _http_get(path, params, timeout=timeout, max_retries=max_retries)
    data = r.json()
    if str(data.get("retCode")) != "0":
        return None
    result = data.get("result") or {}
    lst = result.get("list") or []
    if not lst:
        return None
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
