# EnlilTrading · DataLayer_OHLCV

Назначение: надёжная загрузка, календаризация, ресемплинг и хранение OHLCV-рядов (C1) + автоматические проверки качества и санитайз (C2). Временная зона — UTC.

## Спецификации

* C1: [C1. Data Layer (Постановка задачи).pdf](docs/specs/C1.%20Data%20Layer%20%28Постановка%20задачи%29.pdf)
* C2: [C2. Data Quality (Постановка задачи).pdf](docs/specs/C2.%20Data%20Quality%20%28Потановка%20задачи%29.pdf)
* Сводка: [docs/specs/README.md](docs/specs/README.md)

## Состав

* C1 DataLayer:

  * Fetcher: Bybit v5 `1m` OHLCV с идемпотентным догрузом.
  * Calendarize: детерминированное заполнение внутренних «дыр» плоскими минутами с флагом `is_gap`.
  * Resampler: агрегаты `5m/15m/1h` c прокидыванием `is_gap`.
  * Хранилище: Parquet (ZSTD), метаданные файла, атомарная перезапись.
* C2 DataQuality:

  * Валидатор `validate(df) -> (df_clean, issues)`; детерминированные правки безопасных дефектов.
  * Журнал проблем в CSV; опциональные колонки `dq_flags`, `dq_notes` в данных.

## Зависимости

Python ≥ 3.10. См. `pyproject.toml` или `requirements.txt`.

## Переменные окружения

```
BYBIT_API_KEY=<read-only>
BYBIT_API_SECRET=<secret>
C1_DATA_ROOT=./data            # опционально; иначе используется ./data
C1_BUILD_SIGNATURE=local       # опционально; попадает в метаданные Parquet
```

## Формат данных

Parquet: `data/{SYMBOL}/{TF}.parquet`

Схема столбцов:

* Обязательные: `ts`, `o`, `h`, `l`, `c`, `v`
* Опциональные: `t` (turnover), `is_gap` (bool), `dq_flags` (int32), `dq_notes` (str)

Требования:

* `ts`: тип `timestamp[ns, UTC]`, правая граница бара (минуты кратны).
* Индекс при обработке в памяти — `DatetimeIndex` UTC (правая граница).
* Метаданные файла (в schema metadata):

  * `source=bybit`
  * `symbol`, `tf`
  * `generated_at`, `build_signature`
  * `data_hash` — стабильный хэш содержимого

Компрессия: ZSTD, row group ~256k, словари включены.

## Команды CLI

Модуль: `python -m ohlcv.cli <cmd> ...`

### 1) История 1m (fetch + календаризация + запись)

Спот:

```
python -m ohlcv.cli backfill --symbols BTCUSDT,ETHUSDT --since 2024-01-01 \
  --category spot --data-root ./data
```

Фьючерсы (учёт даты листинга, смещение старта):

```
python -m ohlcv.cli backfill --symbols BTCUSDT --since 2024-01-01 \
  --category linear --spot-align-futures --data-root ./data
```

Поведение:

* Прогресс с оценкой скорости и ETA.
* Heartbeat в «пустых» окнах.
* После загрузки: календаризация 1m, валидация индекса, запись Parquet (идемпотентная).

### 2) Догрузка хвоста 1m

```
python -m ohlcv.cli update --symbols BTCUSDT,ETHUSDT \
  --category spot --data-root ./data
```

### 3) Ресемплинг 1m → 5m/15m/1h

```
python -m ohlcv.cli resample --symbols BTCUSDT --from-tf 1m --to-tf 1h \
  --data-root ./data
```

Правила агрегирования:

* Окно: `label="right", closed="left"`
* `o=first, h=max, l=min, c=last, v=sum, t=sum?`
* `is_gap=max` (если присутствует)

### 4) Отчёт пропусков

```
python -m ohlcv.cli report-missing --symbols BTCUSDT,ETHUSDT \
  --tf 1m --out ./reports/missing.csv --data-root ./data
```

Выход: CSV со сводными метриками пропусков по символам.

### 5) C2 DataQuality: валидация и санитайз

```
python -m ohlcv.cli quality-validate --symbols BTCUSDT,ETHUSDT \
  --tf 1m --write --issues ./reports/issues.csv \
  --miss-fill-threshold 0.0001 --spike-window 200 --spike-k 12.0 --flat-streak 300 \
  --data-root ./data
```

* Записывается очищенный Parquet (если `--write`) и журнал `issues.csv` (append).
* Правки: типы, сортировка, снятие дублей, выравнивание таймстампов, OHLC-инварианты, запрет `v<0`, допустимое заполнение пропусков 1m.
* Журналируются аномалии (скачки цены MAD, длинные серии `v=0`).
* Обновляются `dq_flags`, `dq_notes`.

## Тесты

* C1: `tests/test_resample.py`, `tests/test_store.py`.
* C2: `tests/test_quality.py`.

## Структура проекта

```
ohlcv/
  api/bybit.py              # REST v5: 1m kline, launchTime, heartbeat
  core/validate.py          # проверка/календаризация 1m
  core/resample.py          # ресемплинг 1m → 5m/15m/1h
  io/parquet_store.py       # запись/слияние Parquet, метаданные
  utils/timeframes.py       # мэппинг таймфреймов
  cli.py                    # команды C1/C2
  quality/                  # C2: issues.py, validator.py
  quality/README.md
  version.py
tests/
  test_resample.py
  test_store.py
  test_quality.py
docs/
  howto.md
  specs/
    README.md
    C1. Data Layer (Постановка задачи).pdf
    C2. Data Quality (Потановка задачи).pdf
```

## Ограничения и NFR

* Пропуски < 0.01% внутри диапазона файла 1m.
* Валидность таймстампов (UTC, кратны минуте).
* Задержка обновления ≤ 1 бар.
* Идемпотентная догрузка истории.

## Примеры путей

```
./data/BTCUSDT/1m.parquet
./data/BTCUSDT/5m.parquet
./reports/missing.csv
./reports/issues.csv
```

## Лицензия

MIT
