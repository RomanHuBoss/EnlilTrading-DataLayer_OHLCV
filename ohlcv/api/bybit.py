# ohlcv/api/bybit.py
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

MINUTE_MS = 60_000


@dataclass(frozen=True)
class FetchStats:
    symbol: str
    category: str
    rows: int
    start_ms: int
    end_ms: int
    requests: int
    retries: int
    rate_limited: int


class BybitClient:
    """Минимальный HTTP‑клиент для публичных OHLCV Bybit v5.

    Основной метод: fetch_ohlcv_1m(symbol, start_ms, end_ms, ...)
    Возвращает (DataFrame, FetchStats).
    """

    def __init__(
        self,
        base_url: str = "https://api.bybit.com",
        *,
        timeout: int = 30,
        session: Optional[requests.Session] = None,
        default_category: str = "linear",
        max_retries: int = 5,
        jitter: float = 0.2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.s = session or requests.Session()
        self.default_category = default_category
        self.max_retries = max_retries
        self.jitter = float(jitter)

    # -------------------------
    # Низкоуровневый запрос c backoff
    # -------------------------
    def _sleep(self, seconds: float) -> None:
        time.sleep(max(0.0, seconds))

    def _backoff_seconds(self, attempt: int, base: float = 0.5, cap: float = 60.0) -> float:
        expo = base * (2.0 ** attempt)
        jitter = random.uniform(-self.jitter, self.jitter) * expo
        return float(min(cap, expo + jitter))

    def _request(self, path: str, params: Dict[str, object]) -> Tuple[requests.Response, int, bool]:
        url = f"{self.base_url}{path}"
        tries = 0
        rate_limited = 0
        while True:
            tries += 1
            try:
                resp = self.s.get(url, params=params, timeout=self.timeout)
            except requests.RequestException:
                if tries > self.max_retries:
                    raise
                self._sleep(self._backoff_seconds(tries))
                continue

            status = resp.status_code
            if status == 429:
                rate_limited += 1
                retry_after = resp.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else self._backoff_seconds(tries)
                if tries > self.max_retries:
                    resp.raise_for_status()
                self._sleep(delay)
                continue
            if 500 <= status < 600:
                if tries > self.max_retries:
                    resp.raise_for_status()
                self._sleep(self._backoff_seconds(tries))
                continue
            if status >= 400:
                # 4xx — фатальная ошибка параметров/символа
                resp.raise_for_status()
            return resp, tries, bool(rate_limited)

    # -------------------------
    # Нормализация ответа v5/market/kline
    # -------------------------
    @staticmethod
    def _parse_kline_rows(rows: Iterable[Iterable[object]]) -> pd.DataFrame:
        # Формат по документации v5: [start, open, high, low, close, volume, turnover]
        col_names = ["start", "open", "high", "low", "close", "volume", "turnover"]
        df = pd.DataFrame(list(rows), columns=col_names)
        # приведение типов
        for c in ["open", "high", "low", "close", "volume", "turnover"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["start"] = pd.to_numeric(df["start"], errors="coerce").astype("int64")
        # сортировка по возрастанию времени и удаление дубликатов
        df = df.dropna(subset=["start"]).drop_duplicates(subset=["start"], keep="last").sort_values("start")
        # фильтрация на ровные минуты
        df = df[(df["start"] % MINUTE_MS) == 0]
        # построение индексированного OHLCV
        idx = pd.to_datetime(df["start"], unit="ms", utc=True)
        out = pd.DataFrame(index=idx)
        out.index.name = "ts"
        out["o"] = df["open"].astype("float64")
        out["h"] = df["high"].astype("float64")
        out["l"] = df["low"].astype("float64")
        out["c"] = df["close"].astype("float64")
        out["v"] = df.get("volume", 0.0).astype("float64")
        if "turnover" in df.columns:
            out["t"] = df["turnover"].astype("float64")
        return out

    # -------------------------
    # Публичный метод выгрузки 1m
    # -------------------------
    def fetch_ohlcv_1m(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        *,
        category: Optional[str] = None,
        limit: int = 1000,
        slice_minutes: int = 1_000,  # запас по окну; 1000 минут ≈ 16.6 часов
    ) -> Tuple[pd.DataFrame, FetchStats]:
        """Выгружает минутные свечи [start_ms, end_ms) c пагинацией и backoff.

        Ограничение Bybit: limit<=1000. Для стабильности режем диапазон по кускам slice_minutes,
        подбирая end для каждого запроса.
        """
        if end_ms <= start_ms:
            empty = pd.DataFrame(columns=["o", "h", "l", "c", "v"], index=pd.DatetimeIndex([], name="ts", tz="UTC"))
            return empty, FetchStats(symbol, category or self.default_category, 0, start_ms, end_ms, 0, 0, 0)

        cat = (category or self.default_category)
        all_parts: List[pd.DataFrame] = []
        total_requests = 0
        total_retries = 0
        rate_limited = 0

        cursor = int(start_ms)
        max_span_ms = int(slice_minutes) * MINUTE_MS
        if limit <= 0 or limit > 1000:
            limit = 1000

        while cursor < end_ms:
            window_end = min(end_ms, cursor + max_span_ms)
            params = {
                "category": cat,           # "linear" | "inverse" | "spot"
                "symbol": symbol,
                "interval": "1",
                "start": cursor,
                "end": window_end - 1,    # правую границу делаем включительной
                "limit": int(limit),
            }

            resp, tries, rl = self._request("/v5/market/kline", params)
            total_requests += 1
            total_retries += max(0, tries - 1)
            rate_limited += int(rl)

            data = resp.json()
            if not isinstance(data, dict) or "result" not in data:
                raise RuntimeError("некорректный ответ Bybit: отсутствует поле 'result'")
            result = data.get("result") or {}
            rows = result.get("list") or result.get("kline", [])

            part = self._parse_kline_rows(rows)
            if not part.empty:
                # фильтр по [cursor, window_end)
                mask = (part.index.view("int64") // 1_000_000 >= cursor) & (part.index.view("int64") // 1_000_000 < window_end)
                part = part.loc[mask]
                all_parts.append(part)
                # сдвиг курсора на последнюю минуту + 60_000
                last_ts_ms = int(part.index[-1].value // 1_000_000)
                cursor = max(cursor + MINUTE_MS, last_ts_ms + MINUTE_MS)
            else:
                # пустой блок — маленький шаг вперёд, чтобы не залипать
                cursor = min(end_ms, cursor + limit * MINUTE_MS)

        if all_parts:
            out = pd.concat(all_parts).sort_index()
            out = out[~out.index.duplicated(keep="last")]
        else:
            out = pd.DataFrame(columns=["o", "h", "l", "c", "v"], index=pd.DatetimeIndex([], name="ts", tz="UTC"))

        stats = FetchStats(
            symbol=symbol,
            category=cat,
            rows=int(len(out)),
            start_ms=int(start_ms),
            end_ms=int(end_ms),
            requests=int(total_requests),
            retries=int(total_retries),
            rate_limited=int(rate_limited),
        )
        return out, stats
