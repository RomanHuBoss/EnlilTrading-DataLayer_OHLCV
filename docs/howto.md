# HOWTO · DataLayer_OHLCV (C1 + C2)

Цель: поднять окружение, загрузить 1m, календаризовать, ресемплировать 5m/15m/1h, прогнать DataQuality, выписать отчёты. Все времена — UTC.

**Спецификации:** см. [specs/README.md](./specs/README.md) и оригиналы:
[C1 PDF](./specs/C1.%20Data%20Layer%20%28Постановка%20задачи%29.pdf),
[C2 PDF](./specs/C2.%20Data%20Quality%20%28Потановка%20задачи%29.pdf)

## 1) Создать окружение и установить зависимости

```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -U pip
pip install -e .
```

## 2) Определить переменные окружения

Создай `.env` в корне:

```
BYBIT_API_KEY=...      # read-only
BYBIT_API_SECRET=...
C1_DATA_ROOT=./data    # опционально; по умолчанию ./data
C1_BUILD_SIGNATURE=local
```

`.env.example` — образец.

## 3) Проверка версии и зависимостей

```bash
python -c "import ohlcv, sys; print(ohlcv.__version__); import pyarrow, pandas; print('ok')"
pytest -q
```

Ожидаемо: тесты C1/C2 зелёные.

## 4) Исторический backfill 1m (spot)

```bash
python -m ohlcv.cli backfill \
  --symbols BTCUSDT,ETHUSDT \
  --since 2024-01-01 \
  --category spot \
  --data-root ./data
```

Поведение: прогресс, heartbeat в «пустых» окнах, календаризация 1m (`is_gap`), валидация индекса, запись Parquet (ZSTD, метаданные, идемпотентная догрузка).

## 5) Исторический backfill 1m (futures, выравнивание датой листинга)

```bash
python -m ohlcv.cli backfill \
  --symbols BTCUSDT \
  --since 2020-01-01 \
  --category linear \
  --spot-align-futures \
  --data-root ./data
```

Эффект `--spot-align-futures`: для spot смещает `since` к `min(launchTime(linear), launchTime(inverse))`, чтобы не сканировать пустоту до листинга.

## 6) Догрузка хвоста 1m

```bash
python -m ohlcv.cli update \
  --symbols BTCUSDT,ETHUSDT \
  --category spot \
  --data-root ./data
```

Берёт «последний бар + 1m» как старт. Идемпотентное слияние.

## 7) Ресемплинг 1m → 5m/15m/1h

```bash
python -m ohlcv.cli resample \
  --symbols BTCUSDT \
  --from-tf 1m \
  --to-tf 1h \
  --data-root ./data
```

Правила: окно `label="right", closed="left"`, агрегаты `o=first, h=max, l=min, c=last, v=sum, t=sum?`, `is_gap=max`.

## 8) DataQuality: валидация и санитайз (C2)

```bash
python -m ohlcv.cli quality-validate \
  --symbols BTCUSDT,ETHUSDT \
  --tf 1m \
  --write \
  --issues ./reports/issues.csv \
  --miss-fill-threshold 0.0001 \
  --spike-window 200 \
  --spike-k 12.0 \
  --flat-streak 300 \
  --data-root ./data
```

Выход: перезаписанные очищенные Parquet и `reports/issues.csv` (append). В данных опционально обновляются `dq_flags`, `dq_notes`.

## 9) Отчёт пропусков

```bash
python -m ohlcv.cli report-missing \
  --symbols BTCUSDT,ETHUSDT \
  --tf 1m \
  --out ./reports/missing.csv \
  --data-root ./data
```

CSV с метриками: всего/присутствует/пропущено/доля.

## 10) Схема и метаданные Parquet

Проверка:

```python
import pyarrow.parquet as pq
t = pq.read_table("data/BTCUSDT/1m.parquet")
print(t.schema.metadata)
```

Ожидаемо: `c1.meta` с `symbol, tf, source=bybit, generated_at, build_signature, data_hash`.

## 11) Идемпотентность записи

Повтори шаги 6–8. Итоговый `data_hash` в метаданных не должен зависеть от порядка догрузок при одинаковых данных.

## 12) Минимальные гарантии DoD

* Пропуски < 0.01% внутри диапазона файла 1m.
* Таймстампы кратны целевому ТФ и в UTC.
* Задержка обновления ≤ 1 бар (при запуске `update` по расписанию).
* Все тесты C1/C2 зелёные.

## Пути и схемы

```
./data/{SYMBOL}/1m.parquet
./data/{SYMBOL}/5m.parquet
./data/{SYMBOL}/15m.parquet
./data/{SYMBOL}/1h.parquet
./reports/issues.csv
./reports/missing.csv
```

Колонки: `ts, o, h, l, c, v` + опционально `t, is_gap, dq_flags, dq_notes`.
