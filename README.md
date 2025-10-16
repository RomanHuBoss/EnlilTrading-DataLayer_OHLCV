# DataLayer_OHLCV

Цель: C1 (DataLayer.OHLCV) и C2 (DataQuality) по постановкам от 2025‑10‑15.

## Возможности

* Загрузка 1m OHLCV с Bybit v5 (spot/linear/inverse), устойчивые ретраи и лимиты.
* Идемпотентная запись Parquet per (symbol, tf) c метаданными и детерминированным `data_hash`.
* Ресемплинг из 1m → 5m/15m/1h (`label=right`, `closed=left`).
* Репарация минутных пропусков (`is_gap`), базовая валидация индекса/календаря.
* DataQuality: `validate(df) -> (df_clean, issues)`; флаги `dq_flags`, заметки `dq_notes`.
* CLI: backfill, update, resample, quality-validate, report-missing.

## Установка

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .[dev]
```

Переменные окружения (опционально): `BYBIT_API_KEY`, `BYBIT_API_SECRET`. Каталог данных: `C1_DATA_ROOT` или `./data` по умолчанию.

## CLI

```bash
# История 1m (UTC)
python -m ohlcv.cli backfill --symbols BTCUSDT,ETHUSDT --since 2024-01-01 --category spot --spot-align-futures

# Обновление хвоста до «текущее время - 1 бар»
python -m ohlcv.cli update --symbols BTCUSDT,ETHUSDT --category spot

# Ресемплинг 1m → 1h
python -m ohlcv.cli resample --symbols BTCUSDT,ETHUSDT --from-tf 1m --to-tf 1h

# Отчёт о пропусках
python -m ohlcv.cli report-missing --symbols BTCUSDT,ETHUSDT --tf 1m --out ./reports/missing_1m.csv

# DataQuality: валидация и выгрузка артефактов
python -m ohlcv.cli quality-validate --symbols BTCUSDT,ETHUSDT --tf 1m --write \
  --issues ./reports/issues.csv --issues-parquet ./reports/issues.parquet \
  --quality-summary-csv ./reports/quality_summary.csv --quality-summary-json ./reports/quality_summary.json --truncate
```

## Формат данных

Parquet per `(symbol, tf)`; обязательные колонки: `ts, o, h, l, c, v`; опционально: `t, is_gap, dq_flags, dq_notes`.
Метаданные в `c1.meta` (JSON): `symbol, tf, source, generated_at, build_signature, data_hash, schema`.

## Артефакты DoD/NFR

* Parquet: `data/{symbol}/{1m,5m,15m,1h}.parquet`
* Журнал DQ: `reports/issues.{csv,parquet}`
* Сводка качества: `reports/quality_summary.{csv,json}`
* Отчёты пропусков: `reports/missing_{1m,5m,15m,1h}.csv`
* Логи интеграции: `docs/releases/<DATE>/integration.log`
* Проверка задержки: `tools/check_latency.py`

## CI

* Покрытие для `ohlcv/quality`: 100% (`pytest-cov`).
* Гейт DQ: падение при наличии дефектов на данных репозитория.
  Workflow: `.github/workflows/ci.yml`.

## Спецификации

* [C1 Data Layer](docs/specs/C1-Data%20Layer.pdf)
* [C2 Data Quality](docs/specs/C2-Data%20Quality.pdf)
