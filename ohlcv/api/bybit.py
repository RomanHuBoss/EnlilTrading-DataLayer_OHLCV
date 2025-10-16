# ohlcv/api/bybit.py — Bybit v5 REST, публичные эндпоинты для OHLCV 1m и launchTime
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Generator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    TypedDict,
)

import requests

BASE_URL = "https://api.bybit.com"
USER_AGENT = "EnlilTrading-DataLayer/1.0 (python-requests)"

KLINE_LIMIT = 1000  # limit<=1000, интервал 1m → "1"
RATE_LIMIT_CODES = {"10006"}


class KlineRow(TypedDict, total=False):
    ts: str
    o: float
    h: float
    l: float  # noqa: E741 — оставляем ключ 'l' ради совместимости со схемой C1/C2
    c: float
    v: float
    t: float


def _parse_retry_after_seconds(resp: requests.Response) -> Optional[float]:
    h = resp.headers or {}
    for k in ("Retry-After", "retry-after"):
        if k in h:
            try:
                return float(h[k])
            except Exception:
                pass
    now = time.time()
    for k in (
        "X-RateLimit-Reset",
        "X-Rate-Limit-Reset",
        "X-Bapi-Limit-Reset-Timestamp",
        "X-Bapi-Rate-Limit-Reset-Timestamp",
    ):
        if k in h:
            val = h[k]
            try:
                ts = float(val)
                if ts > 10_000_000_000:
                    ts /= 1000.0
                wait = max(0.0, ts - now)
                if wait > 0:
                    return wait
            except Exception:
                pass
    try:
        data = resp.json()
        ext = data.get("retExtInfo") or {}
        for k in ("retryAfter", "waitSec", "nextValidTimestamp"):
            if k in ext:
                v = float(ext[k])
                if v > 10_000_000_000:
                    v /= 1000.0
                if k == "nextValidTimestamp":
                    return max(0.0, v - time.time())
                return max(0.0, v)
    except Exception:
        pass
    return None


def _http_get(
    path: str,
    params: Mapping[str, str | int | float | None],
    headers: Mapping[str, str] | None = None,
    *,
    timeout: int = 20,
    max_retries: int = 10,
    backoff_base: float = 0.4,
) -> requests.Response:
    url = f"{BASE_URL}{path}"
    h: MutableMapping[str, str] = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)

    last_err: Optional[Exception] = None
    sleep = backoff_base

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
            if r.status_code == 429:
                wait = _parse_retry_after_seconds(r) or sleep
                time.sleep(wait)
                sleep = min(sleep * 2, 8.0)
                continue
            try:
                data = r.json()
                rc = str(data.get("retCode"))
                if rc in RATE_LIMIT_CODES:
                    wait = _parse_retry_after_seconds(r) or sleep
                    time.sleep(wait)
                    sleep = min(sleep * 2, 8.0)
                    continue
            except Exception:
                pass
            return r
        except Exception as e:
            last_err = e
            if attempt >= max_retries:
                break
            time.sleep(sleep)
            sleep = min(sleep * 2, 8.0)

    if isinstance(last_err, Exception):
        raise last_err
    raise RuntimeError("HTTP GET failed with unknown error")


def _iso_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _rows_from_kline_list(lst: Sequence[Sequence[str]]) -> List[KlineRow]:
    out: List[KlineRow] = []
    for it in lst:
        start_ms = int(it[0])
        o = float(it[1])
        high = float(it[2])
        low = float(it[3])
        c = float(it[4])
        v = float(it[5])
        ts_iso = _iso_from_ms(start_ms + 60_000)
        row: KlineRow = {"ts": ts_iso, "o": o, "h": high, "l": low, "c": c, "v": v}
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
    sleep_sec: float = 0.25,
) -> Generator[List[KlineRow], None, None]:
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    if until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)
    if since >= until:
        return

    start_ms = int(since.timestamp() * 1000)
    end_ms_initial = int(until.timestamp() * 1000)

    path = "/v5/market/kline"
    params_base: dict[str, str | int] = {
        "category": category,
        "symbol": symbol,
        "interval": "1",
        "limit": KLINE_LIMIT,
    }

    window_end = end_ms_initial
    step_ms = KLINE_LIMIT * 60_000

    while window_end > start_ms:
        params: dict[str, int | str] = dict(params_base)
        params["start"] = start_ms
        params["end"] = window_end

        r = _http_get(
            path, params, timeout=timeout, max_retries=10, backoff_base=max(sleep_sec, 0.25)
        )
        data: dict[str, Any] = r.json()
        if str(data.get("retCode")) != "0":
            raise RuntimeError(f"Bybit error: {data.get('retCode')} {data.get('retMsg')}")
        result = data.get("result") or {}
        lst = result.get("list") or []

        if not lst:
            new_end = max(start_ms, window_end - step_ms)
            if new_end == window_end:
                break
            window_end = new_end
            if on_advance:
                on_advance(window_end, end_ms_initial)
            time.sleep(sleep_sec)
            continue

        lst_sorted = sorted(lst, key=lambda x: int(x[0]))
        rows = _rows_from_kline_list(lst_sorted)
        yield rows

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
) -> List[KlineRow]:
    acc: List[KlineRow] = []
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
    path = "/v5/market/instruments-info"
    params: Mapping[str, str] = {"category": category, "symbol": symbol}
    r = _http_get(path, params, timeout=timeout, max_retries=max_retries)
    data: dict[str, Any] = r.json()
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
