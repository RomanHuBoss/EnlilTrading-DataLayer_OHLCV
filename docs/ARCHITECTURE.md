# Архитектура: C1 / C2 / C3

## Обзор

Пайплайн единого репозитория:

```
C1 · DataLayer.OHLCV  →  C2 · DataQuality  →  C3 · Features.Core
   (импорт/норм.)         (валидация/санитайз)    (детермин. признаки)
```

## Компоненты

### C1 · DataLayer.OHLCV

* Источники: Bybit v5 (spot/linear/inverse).
* Артефакты: Parquet per `(symbol, tf)`.
* Ресемплинг 1m→{5m,15m,1h}: `label=right`, `closed=left`.
* Поля: `ts, o, h, l, c, v` (+ `t?`, `is_gap?`, `dq_flags?`, `dq_notes?`).

### C2 · DataQuality

* Валидация индекса/календаря, заполненность, спайки/флэты.
* Санитайз подготовленных файлов (перезапись Parquet).
* Отчёты: `reports/issues.{csv,parquet}`, `reports/quality_summary.{csv,json}`.

### C3 · Features.Core

* Вход: валидированные бары из C2.
* Выход: исходные поля + `f_*`, `symbol`, `tf`, `f_valid_from`, `f_build_version`.
* CLI: `features-core build`.

## Контракты данных

### Вход C3 (канон)

* Колонки обязательные: `timestamp_ms:int64`, `start_time_iso:str(UTC)`, `open:float64`, `high:float64`, `low:float64`, `close:float64`, `volume:float64`.
* Опционально: `turnover:float64`.
* Допускается внутренняя альтернативная схема C1/C2: `o,h,l,c,v,(t?)` — автопереименование.

### Выход C3

* Новые признаки `f_*` (см. `docs/C3_FEATURES.md`).
* Служебные: `symbol:str`, `tf:str`, `f_valid_from:int`, `f_build_version:str`.
* Форматы: Parquet/CSV.

## Инварианты и численные правила

* Rolling: `min_periods = window`.
* `std(ddof=1)`.
* Эпсилон: `ε = 1e-12` в делителях.
* Wilder-сглаживание: `ewm(alpha=1/n, adjust=False)` для RSI/ATR/ADX.

## Версионирование сборок

* `f_build_version = C3.Features.Core@<semver>+<hash12>`.
* Хэш по git-ревизии и параметрам окон.

## Потоки и I/O

* C1/C2: читают/пишут в `./data` и `./reports` (конфигурируемо).
* C3: читает CSV/Parquet входа, пишет Parquet/CSV с признаками.
* Сетевых вызовов в C3 нет.

## Ошибки и строгий режим

* Нормализация/валидация входа: `ohlcv/features/schema.py`.
* CLI флаг `--strict`: падение при нарушении схемы/NaN в базовых колонках.

## Производительность

* Векторизация `pandas`.
* Параметры окон по умолчанию: `rv {20,60}`, `donch 20`, `z {20,60}`, `vwap_roll 96`, `rsi/adx/atr 14`, `ema/mom 20`.

## Тестирование

* Unit: расчёты признаков, `f_valid_from`.
* Strict: отказ при NaN.
* CLI-интеграция: `features-core build` на маленьком CSV/Parquet.

## CI

* Матрица Python 3.11/3.12.
* Линтинг/типизация: ruff, black, isort, mypy.
* Покрытие: отчёт по `ohlcv` с минимумом для `ohlcv.features`.
