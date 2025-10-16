# OHLCV Pipeline — C1 / C2 / C3

Назначение: единый репозиторий для трёх компонентов пайплайна по временным рядам рынка:
C1 — загрузка и нормализация OHLCV,
C2 — контроль качества и подготовка валидированных артефактов,
C3 — расчёт детерминированных признаков поверх валидированных баров.

## Состав компонентов

- **C1 · DataLayer.OHLCV** — импорт, нормализация, ресемплинг, идемпотентная запись Parquet, отчёты пропусков.
- **C2 · DataQuality** — валидация, санитайз, сводные отчёты качества, флаги DQ на данных.
- **C3 · Features.Core** — вычисление признаков из валидированных OHLCV; без внешних TA-зависимостей; pandas-only.

## Требования

- Python 3.11+
- pandas >= 2.2 (требуется для C3; совместимо с C1/C2)
- Остальные зависимости — см. `requirements.txt`

## CLI (сводка)

## C1/C2

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

```bash
features-core build   --input path/to/ohlcv.csv   --symbol BTCUSDT   --tf 5m   --config configs/features.example.yaml   --output out/BTCUSDT_5m_features.parquet
```

## C3 · Features.Core

### Вход/выход
- Вход: CSV/Parquet из C2 с колонками `timestamp_ms,start_time_iso,open,high,low,close,volume,(turnover?)`; допускается внутренняя схема `o,h,l,c,v,(t?)` (автопереименование).
- Выход: исходные поля + признаки `f_*`, а также `symbol`, `tf`, `f_valid_from`, `f_build_version`.

### Набор признаков по умолчанию
- Доходности/вола: `f_ret1`, `f_logret1`, `f_rv_20`, `f_rv_60`
- Свечные: `f_range_pct`, `f_body_pct`, `f_wick_upper_pct`, `f_wick_lower_pct`, `f_tr`, `f_atr_14`, `f_atr_pct_14`
- Тренд/моментум: `f_ema_20`, `f_ema_slope_20`, `f_mom_20`, `f_rsi14`, `f_pdi14`, `f_mdi14`, `f_adx14`
- Donchian 20: `f_donch_h_20`, `f_donch_l_20`, `f_donch_break_dir_20`, `f_donch_width_pct_20`
- Z-score: `f_close_z_20`, `f_close_z_60`, `f_range_z_20`, `f_range_z_60`, `f_vol_z_20`, `f_vol_z_60`
- Объёмы: `f_upvol_20`, `f_downvol_20`, `f_vol_balance_20`, `f_obv`
- VWAP: `f_vwap_roll_96`, `f_vwap_dev_pct_96`, `f_vwap_session`, `f_vwap_session_dev_pct`

### Версионирование сборок признаков
`f_build_version = C3.Features.Core@0.1.0+<hash12>`; хэш учитывает git-ревизию и параметры.

---

## CI

* Покрытие для `ohlcv/quality`: 100% (`pytest-cov`).
* Гейт DQ: падение при наличии дефектов на данных репозитория.
  Workflow: `.github/workflows/ci.yml`.

## Спецификации

* [C1 Data Layer](docs/specs/C1-Data%20Layer.pdf)
* [C2 Data Quality](docs/specs/C2-Data%20Quality.pdf)
* [C3 Features](docs/specs/C3-Features.pdf)
