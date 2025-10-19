# Проект: Маркет‑данные и режимы рынка

Комплект модулей для загрузки минутных OHLCV (Bybit), очистки и выравнивания, ресемплинга в целевые ТФ, построения признаков и (опционально) детекции рыночных режимов.

Состав:
- **C1 · Data Layer (OHLCV)** — загрузка 1m по Bybit, нормализация, календаризация, Parquet‑хранилище, ресемплинг 1m→5m/15m/1h, отчёты по пропускам, DataReader.
- **C2 · Data Quality** — инварианты OHLC, выравнивание к минутному календарю, флаг `is_gap`, агрегированные отчёты и порог для CI. Реализовано внутри `ohlcv.core.validate` и CLI `report-missing` (см. ниже).
- **C3 · Features** — минимальный согласованный набор признаков по постановке: волатильность, тренд/импульс, свечные относительные метрики, VWAP, объёмные показатели, Donchian и т.п. CLI `c3-features`.
- **C4 · Regime (опционально)** — ансамбль детекторов (ADX/Donchian/BOCPD/HMM). Пакет в исходниках, сборка в дистрибутив не включена.

---
## Требования
- Python ≥ 3.9
- `numpy`, `pandas`, `pyarrow`, `requests` (см. `pyproject.toml`)

Установка для разработки:
```bash
pip install -e .[test]
```

---
## Быстрый старт
### 1) Загрузка минутных данных (C1)
```bash
c1-ohlcv backfill \
  --symbol BTCUSDT \
  --store ./data \
  --since-ms 1700000000000 \
  --until-ms 1700003600000
```
Выход: `./data/BTCUSDT/1m.parquet` с метаданными `c1.meta` в footer.

Догрузка хвоста:
```bash
c1-ohlcv update --symbol BTCUSDT --store ./data
```

Ресемплинг 1m→5m:
```bash
c1-ohlcv resample --symbol BTCUSDT --store ./data --dst-tf 5m
```

Отчёт по пропускам (C2):
```bash
c1-ohlcv report-missing \
  --symbol BTCUSDT --store ./data \
  --since-ms 1700000000000 --until-ms 1700003600000 \
  --fail-gap-pct 5
```
Код возврата `2` при превышении порога пропусков.

### 2) Построение признаков (C3)
```bash
c3-features build \
  input.parquet \
  features.parquet \
  --symbol BTCUSDT --tf 5m \
  --config configs/features.example.yaml
```
Выход содержит ключевые столбцы: `f_rv_*, f_adx_14, f_pdi_14, f_mdi_14, f_donch_width_pct_*, f_rsi_14, f_vwap_*`, свечные `%`‑метрики, `f_valid_from`, `f_build_version`.

---
## Структура репозитория
```
ohlcv/
  api/bybit.py            # клиент Bybit v5 (публичный), backoff+джиттер
  cli.py                  # CLI: backfill, update, resample, read, report-missing
  core/validate.py        # нормализация 1m, календарь, is_gap
  core/resample.py        # агрегация 1m→{5m,15m,1h}
  io/parquet_store.py     # Parquet: идемпотентная запись, метаданные c1.meta
  io/tail_cache.py        # sidecar <symbol>/<tf>.latest.json
  datareader.py           # DataReader для диапазонов и дней
  utils/timeframes.py     # парсинг/выравнивание таймфреймов
  version.py              # __version__, build_signature()

ohlcv/features/
  __init__.py, core.py, schema.py, cli.py, utils.py

c4_regime/                # ансамбль детекторов (опционально)

configs/
  features.example.yaml   # пример конфигурации C3

tests/
  test_*.py               # юнит‑ и интеграционные тесты
```

---
## Схемы данных
### Паркет‑хранилище (C1)
- Путь: `<root>/<symbol>/<tf>.parquet`
- Индекс‑колонка: `ts` (tz‑aware, UTC, timestamp[ns])
- Данные: `o,h,l,c,v` (float64), опционально `t` (turnover, float64), `is_gap` (bool)
- Сжатие: `zstd(level=7)`; группировка: `row_group_size=256_000`; `use_dictionary=True`
- Метаданные (footer, ключ `c1.meta`):
  ```json
  {
    "symbol": "BTCUSDT",
    "tf": "1m",
    "rows": 123456,
    "min_ts": "2024-01-01T00:00:00+00:00",
    "max_ts": "2024-01-02T00:00:00+00:00",
    "generated_at": "<UTC ISO>",
    "data_hash": "<sha1>",
    "build_signature": "C1-0.1.0+p<psha>+g<sha>[-dirty]",
    "schema": ["ts","o","h","l","c","v","t","is_gap"],
    "schema_version": 1,
    "zstd_level": 7,
    "row_group_size": 256000
  }
  ```

### Features (C3)
- Вход: OHLCV с колонками `timestamp_ms,start_time_iso,open,high,low,close,volume` или паркет с индексом `ts`.
- Выход: признаки по постановке C3. Прогрев `f_valid_from` = максимум окон.
- Версионирование: `f_build_version = C3-<ver>+p<psha>(+g<sha>[-dirty])`.

---
## DataReader
Высокоуровневое чтение диапазонов и суток:
```python
from ohlcv import DataReader
r = DataReader("./data")
# выборка с parquet‑filters
df, st = r.read_range("BTCUSDT", "1m", start_ms=1700000000000, end_ms=1700003600000, columns=["o","c"]) 
# выравнивание к минутному календарю + флаг is_gap
aligned, st2 = r.read_range("BTCUSDT", "1m", start_ms=..., end_ms=..., align_1m=True)
# сутки UTC
day_df, _ = r.read_day_utc("BTCUSDT", "5m", "2024-01-01")
```
Sidecar‑кэш: `<root>/<symbol>/<tf>.latest.json` с полями `latest_ts_ms, rows_total, data_hash, updated_at`.

---
## CI/Качество (C2)
- Проверка непрерывности минутных баров: `c1-ohlcv report-missing ... --fail-gap-pct <THRESHOLD>` → код `2` при превышении.
- Инварианты: `high>=low`, `open/close ∈ [low,high]`. При `strict=True` нарушения вызывают ошибку.

---
## Примеры пайплайнов
### От сырого 1m к признакам 5m
```bash
# 1) минутные бары
c1-ohlcv backfill --symbol BTCUSDT --store ./data --since-ms $SINCE --until-ms $UNTIL
# 2) ресемплинг 1m→5m
c1-ohlcv resample --symbol BTCUSDT --store ./data --dst-tf 5m
# 3) признаки 5m
c3-features build ./data/BTCUSDT/5m.parquet ./data/BTCUSDT/features_5m.parquet --symbol BTCUSDT --tf 5m
```

---
## Переменные окружения
- `GIT_COMMIT` / `CI_COMMIT_SHA` / `SOURCE_VERSION` — используются для формирования `build_signature`.

---
## Тесты
```bash
pytest -q
```

---
## Траблшутинг
- `pyarrow` отсутствует → Parquet‑операции недоступны. Установить зависимость.
- `ValueError: ожидался столбец 'ts'` при чтении — повреждён файл/неверный формат; перезаписать через `write_idempotent`.
- Пустые ответы Bybit на свежем окне — возможная пауза/делист; повторить с меньшим интервалом.

---
## Версионирование
- `ohlcv/version.py::__version__` — версия пакета.
- `build_signature(params, component)` — сигнатура сборки: `C1-<ver>+p<psha>+g<sha>[-dirty]`.

---
## Лицензия
Proprietary
