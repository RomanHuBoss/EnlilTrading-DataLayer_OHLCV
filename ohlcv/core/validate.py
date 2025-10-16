# ohlcv/core/validate.py
# Валидация и репарация минутных OHLCV-рядов.
# Назначение:
#   — validate_1m_index: строгая проверка индекса 1m.
#   — missing_rate: расчёт доли пропусков внутри диапазона df.
#   — ensure_missing_threshold: исключение при превышении порога пропусков.
#   — fill_1m_gaps: детерминированная календаризация 1m с пометкой синтетики is_gap.
#
# Допущения:
#   — Индекс входного df — tz-aware UTC DatetimeIndex с правыми границами минутных баров.
#   — Обязательные столбцы: o, h, l, c, v. Опционально: t (turnover).
#   — Все времена — UTC. Частота минутная, без дублей.

from __future__ import annotations

from typing import Tuple
import pandas as pd


def validate_1m_index(df: pd.DataFrame) -> None:
    """
    Строгая валидация индекса для 1m.
    Требования:
      1) DatetimeIndex, tz-aware, UTC.
      2) Монотонность по возрастанию.
      3) Отсутствие дублей.
      4) Выравнивание по минуте (секунды == 0).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Индекс должен быть DatetimeIndex")
    if df.index.tz is None or str(df.index.tz) != "UTC":
        raise ValueError("Индекс должен быть в UTC (tz-aware)")
    if not df.index.is_monotonic_increasing:
        raise ValueError("Индекс должен быть монотонно возрастающим")
    if df.index.duplicated().any():
        raise ValueError("Обнаружены дубликаты таймстампов")
    # кратность минуте: все метки должны приходиться на ровную минуту (секунды==0)
    # вычисление по секундам UNIX (int64):
    if ((df.index.view("i8") // 10**9) % 60).any():
        raise ValueError("Таймстампы не выровнены по минутам")


def missing_rate(df: pd.DataFrame, freq: str = "min") -> float:
    """
    Доля пропусков внутри замкнутого диапазона индекса df относительно полной регулярной сетки.
    freq: 'min' (минуты). Индекс df — UTC.
    """
    if df.empty:
        return 0.0
    full = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
    # количество уникальных меток во входе по отношению к полной сетке
    rate = 1.0 - (len(df.index.unique()) / max(1, len(full)))
    return max(0.0, rate)


def ensure_missing_threshold(df: pd.DataFrame, threshold: float = 0.0001) -> None:
    """
    Исключение при превышении порога доли пропусков.
    По умолчанию 0.01% (0.0001), как в NFR.
    """
    rate = missing_rate(df)
    if rate > threshold:
        raise ValueError(f"Доля пропусков {rate:.6f} превышает порог {threshold:.6f}")


def fill_1m_gaps(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Календаризация 1m с детерминированным заполнением «дыр» синтетическими барами.
    Политика заполнения пропусков:
      — close: forward-fill по предыдущему close; для самого начала — bfill.
      — open:  равен заполненному close (плоская свеча).
      — high/low: равны open/close.
      — volume: 0.0
      — turnover 't' (если присутствует): 0.0
      — is_gap: bool-флаг, True только у синтетических вставок.

    Возвращает:
      (df_filled, n_filled), где n_filled — число синтетических минут.
    """
    if df.empty:
        return df, 0
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError("Ожидается tz-aware DatetimeIndex (UTC)")

    # Полная минутная сетка в границах имеющихся данных
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="min", tz="UTC")

    # Расширяем до полной сетки
    out = df.reindex(full_idx)

    # Маска пропусков (нет исходного бара на эту минуту)
    missing = out["c"].isna()

    if missing.any():
        # Базовый close для заполнения: ffill, затем bfill для случая «дыр» в начале
        base_close = out["c"].ffill().bfill()

        # Заполняем ценовые поля плоской свечой
        out.loc[missing, "c"] = base_close.loc[missing]
        out.loc[missing, "o"] = base_close.loc[missing]
        out.loc[missing, "h"] = base_close.loc[missing]
        out.loc[missing, "l"] = base_close.loc[missing]

        # Объём и оборот — нули для синтетики
        out.loc[missing, "v"] = 0.0
        if "t" in out.columns:
            out.loc[missing, "t"] = 0.0

    # Флаг календаризации
    out["is_gap"] = False
    if missing.any():
        out.loc[missing, "is_gap"] = True

    # Сортировка и возврат
    out = out.sort_index()
    return out, int(missing.sum())
