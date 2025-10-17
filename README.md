# OHLCV Pipeline — C1 / C2 / C3

Назначение: единый репозиторий для трёх компонентов пайплайна по временным рядам рынка:
C1 — загрузка и нормализация OHLCV,
C2 — контроль качества и подготовка валидированных артефактов,
C3 — расчёт детерминированных признаков поверх валидированных баров.

## Состав компонентов

* **C1 · DataLayer.OHLCV** — импорт, нормализация, ресемплинг, идемпотентная запись Parquet, отчёты пропусков.
* **C2 · DataQuality** — валидация, санитайз, сводные отчёты качества, флаги DQ на данных.
* **C3 · Features.Core** — вычисление признаков из валидированных OHLCV; без внешних TA-зависимостей; pandas-only.

## Требования

* Python 3.11+
* pandas ≥ 2.2
* Остальные зависимости — см. `pyproject.toml`

## Установка

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .[dev]
```

Переменные окружения (опционально): `BYBIT_API_KEY`, `BYBIT_API_SECRET`. Каталог данных: `C1_DATA_ROOT` или `./data` по умолчанию.

## CLI (сводка)

### C1/C2

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

### Формат данных

Parquet per `(symbol, tf)`; обязательные колонки: `ts, o, h, l, c, v`; опционально: `t, is_gap, dq_flags, dq_notes`.
Метаданные в `c1.meta` (JSON): `symbol, tf, generated_at, data_hash, schema`.

---

## C3 · Features.Core

```bash
# Пример: формирование признаков на минутных барах
python -m ohlcv.features.cli build \
  --input ./data/BTCUSDT/1m.parquet \
  --symbol BTCUSDT \
  --tf 1m \
  --config configs/features.example.yaml \
  --output out/BTCUSDT_1m_features.parquet
```

### Важное про `--tf` и частоту входа

* Частота признаков определяется частотой **входного** файла.
* Параметр `--tf` в `features.cli` — это **метка** в выходе (`tf`), а не механизм ресемплинга.
* Для часовых признаков подай часовые бары. Ресемплинг выполняется **в C1** (`python -m ohlcv.cli resample`), а не в `features.cli`.

Корректные последовательности:

```bash
# 1) Минутные фичи из минутного входа
python -m ohlcv.features.cli build --input data/SYM/1m.parquet --symbol SYM --tf 1m --output out/SYM_1m_features.parquet

# 2) Часовые фичи: сначала ресемплинг 1m→1h, затем расчёт признаков на 1h
python -m ohlcv.cli resample --symbols SYM --from-tf 1m --to-tf 1h --data-root ./data
python -m ohlcv.features.cli build --input data/SYM/1h.parquet --symbol SYM --tf 1h --output out/SYM_1h_features.parquet

# 3) Если указать --tf 1h на минутном входе — частота признаков останется минутной; изменится только метка tf в выходе
python -m ohlcv.features.cli build --input data/SYM/1m.parquet --symbol SYM --tf 1h --output out/SYM_mislabel.parquet
# Некорректное использование; не полагайся на такую сборку.
```

### Вход/выход

* Вход: CSV/Parquet из C2 с колонками `timestamp_ms,start_time_iso,open,high,low,close,volume,(turnover?)`; допускается внутренняя схема `o,h,l,c,v,(t?)` (автопереименование).
* Выход: исходные поля + признаки `f_*`, а также `symbol`, `tf`, `f_valid_from`, `f_build_version`.

### Набор признаков по умолчанию

* Доходности/вола: `f_ret`, `f_logret`, `f_rv_{20,60}`
* Свечные: `f_tr`, `f_atr_14`, ширина Дончиана
* Тренд/моментум: EMA/SMA, `±DI/ADX`
* Z-score: `f_z_close_{20,60}`
* Объёмы: скользящие средние объёма
* `f_valid_from = max(всех окон)` — позиция первой полной строки без NaN по всем `f_*`

---

## Артефакты

* Parquet: `data/{symbol}/{1m,5m,15m,1h}.parquet`
* Журнал DQ: `reports/issues.{csv,parquet}`
* Сводка качества: `reports/quality_summary.{csv,json}`
* Отчёты пропусков: `reports/missing_{1m,5m,15m,1h}.csv`

## Структура репозитория

```
ohlcv/
  cli.py                # C1/C2: backfill/update/resample/report/quality-validate
  core/                 # базовые операции над рядами (ресемплинг/валидации)
  io/                   # parquet_store и I/O-утилиты
  quality/              # валидатор данных (C2)
  features/             # вычисление признаков и CLI (C3)
configs/
  features.example.yaml # пример конфигурации C3
tests/                  # юнит- и интеграционные тесты
data/                   # корень данных по умолчанию (локально)
reports/                # отчёты (issues, quality_summary, missing)
```

## CI

`pytest -q --maxfail=1 --disable-warnings --cov=ohlcv --cov-report=term-missing`

## Спецификации

* [C1 Data Layer](docs/specs/C1-Data%20Layer.pdf)
* [C2 Data Quality](docs/specs/C2-Data%20Quality.pdf)
* [C3 Features](docs/specs/C3-Features.pdf)
