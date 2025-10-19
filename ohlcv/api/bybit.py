from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import math
import random
import time

import pandas as pd
import requests

__all__ = ["BybitClient", "FetchStats"]


@dataclass
class FetchStats:
    requests: int = 0
    pages: int = 0
    rows: int = 0
    retries: int = 0


class BybitClient:
    """Минимальный клиент Bybit v5 для C1 backfill.

    Использует публичный эндпоинт kline v5. Ключ не требуется, но может быть передан.
    Пагинация реализована временем: следующий чанк начинается после максимального ts.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://api.bybit.com",
        read_only_key: Optional[str] = None,
        timeout_s: int = 10,
        max_retries: int = 5,
        max_concurrent: int = 2,  # зарезервировано
        category: str = "linear",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.read_only_key = read_only_key
        self.timeout_s = int(timeout_s)
        self.max_retries = int(max_retries)
        self.max_concurrent = int(max_concurrent)
        self.category = category
        self.stats = FetchStats()

    # ------------------
    # Внутренний HTTP
    # ------------------
    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "ohlcv-c1/1.0",
        }
        if self.read_only_key:
            h["X-BAPI-API-KEY"] = self.read_only_key
        return h

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        attempt = 0
        while True:
            try:
                self.stats.requests += 1
                r = requests.get(url, params=params, headers=self._headers(), timeout=self.timeout_s)
                if r.status_code == 429:
                    # простой бэкофф по retry-after
                    after = float(r.headers.get("Retry-After", "1"))
                    time.sleep(after)
                    attempt += 1
                    self.stats.retries += 1
                    if attempt > self.max_retries:
                        r.raise_for_status()
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException:
                attempt += 1
                self.stats.retries += 1
                if attempt > self.max_retries:
                    raise
                # экспоненциальный бэкофф с джиттером
                sleep_s = min(60.0, (2 ** min(attempt, 6)) + random.random())
                time.sleep(sleep_s)

    # ------------------
    # Публичные методы
    # ------------------
    def fetch_klines_1m(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        limit: int = 1000,
    ) -> Tuple[pd.DataFrame, Optional[int]]:
        """Получает свечи 1m в [start_ms, end_ms). Возвращает (df, next_ms).

        next_ms — отметка времени, с которой нужно продолжать, либо None, если достигнут конец диапазона.
        """
        if end_ms <= start_ms:
            return pd.DataFrame(columns=["ts", "o", "h", "l", "c", "v", "t"]), None

        # Ограничим окно одним лимитом по минутам
        max_window_ms = limit * 60_000
        local_end = min(end_ms, start_ms + max_window_ms)

        params: Dict[str, Any] = {
            "category": self.category,
            "symbol": symbol,
            "interval": "1",
            "start": int(start_ms),
            # Bybit end — включительно; используем local_end-60_000, чтобы не перехватить лишнюю минуту
            "end": int(local_end - 1),
            "limit": int(limit),
        }
        data = self._get("/v5/market/kline", params=params)

        # Проверка формата ответа
        if int(data.get("retCode", 0)) != 0:
            raise RuntimeError(f"Bybit error: {data.get('retCode')} {data.get('retMsg')}")
        result = data.get("result") or {}
        rows = result.get("list") or []

        if not rows:
            return pd.DataFrame(columns=["ts", "o", "h", "l", "c", "v", "t"]), None

        # list: [start, open, high, low, close, volume, turnover]
        df = pd.DataFrame(rows, columns=["ts", "o", "h", "l", "c", "v", "t"]).astype(str)
        # Приведение типов
        df["ts"] = df["ts"].astype("int64")
        for col in ["o", "h", "l", "c", "v", "t"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        # Сортировка и фильтр на всякий случай
        df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
        df = df[(df["ts"] >= start_ms) & (df["ts"] < end_ms)]

        self.stats.pages += 1
        self.stats.rows += int(len(df))

        if len(df) == 0:
            return df, None

        last_ts = int(df["ts"].max())
        # Если мы исчерпали локальное окно, продолжаем со следующей минуты
        if last_ts >= (local_end - 60_000) and local_end < end_ms:
            next_ms = last_ts + 60_000
        else:
            next_ms = None

        return df, next_ms
