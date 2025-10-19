# OHLCV Data Layer — C1/C2/C3/C4

## Содержание

* Архитектура и контракты (C1–C4)
* Форматы данных и колонок
* CLI команды и коды возврата
* Структура хранилища (Parquet Store)
* Конфиги
* Совместимость и миграция

---

## Архитектура

* **C1 Data Layer**: нормализация 1m, выравнивание минутной сетки, ресемплинг 1m→5m/15m/1h, хвостовой кэш `tail`.
* **C2 Data Quality**: правила валидации (R01–R14, R20–R23, R30–R33), отчёты `issues` в каноне.
* **C3 Features**: стабильные фичи (ATR/RSI/ADX/Donchian/VWAP/Z-score/объёмные), мета‑поля и строгая схема.
* **C4 Regime**: ансамблевый режим рынка (ADX‑тренд, Donchian‑прорыв, changepoint по z‑score, HMM‑proxy RV).

Пакеты:

```
ohlcv.core         # C1: normalize/align/resample
ohlcv.io           # C1: parquet store + tail
ohlcv.api          # C1: Bybit v5 клиент
ohlcv.quality      # C2: validator + issues
ohlcv.features     # C3: build + schema
ohlcv.regime       # C4: infer_regime
```

## Форматы данных

### Канон 1m и агрегатов

* Индекс/поле времени: `ts` — правая граница бара, миллисекунды UTC (`int64`).
* Колонки: `o,h,l,c,v[,t],is_gap` (`float64`, `is_gap: bool`).
* Пропуски минут заполняются **синтетическим плоским баром** (`o=h=l=c=prev_close`, `v=0`, `is_gap=true`).

### `issues` (C2)

Столбцы: `ts,code,note,severity,action,dq_rank,symbol,tf`.

* Коды: R01–R14, R20–R23, R30–R33 (+ служебные). Алиасы приводятся к канону.

### Фичи (C3)

`ts:int64, symbol:str, tf:str, f_valid_from:int64, f_build_version:str, f_*:float64`.

### Режим (C4)

`ts:int64, regime:{-1,0,1}, regime_name:{bear,calm,bull}, score:int, d1_*, d2_*, d3_*, d4_*`.

## CLI

Двоичный скрипт: `ohlcv`.

### C1

* `ohlcv datalayer resample SRC.parquet --dst-tf {5m|15m|1h} --out OUT.parquet [--compression zstd]`
* `ohlcv datalayer missing-report SRC_1m.parquet [--since-ms MS] [--until-ms MS] [--fail-gap-pct 0..1]`

  * Возврат `2`, если доля пропусков > порога.
* `ohlcv datalayer backfill SYMBOL --store STORE_DIR --since-ms MS --until-ms MS [параметры API]`

Совместимые алиасы:

* `ohlcv c1-resample`, `ohlcv c1-backfill`, `ohlcv c1-report-missing`, `ohlcv c1-update`.

### C2

* `ohlcv dataquality validate IN.parquet [--tf TF] [--symbol S] [--ref-1m REF.parquet] [--out-dir DIR] [--quality-json OUT.json] [--write-inplace] [--cfg-json CFG.json]`

  * Пишет очищенный parquet и `*.issues.parquet`.

## Структура хранилища

```
<STORE>/<SYMBOL>/
  1m.parquet
  5m.parquet
  15m.parquet
  1h.parquet
  tail/
    1m.tail.json
    1m.tail.parquet
```

* Запись идемпотентная: мердж по `ts` (последний выигрывает), атомарная.
* Метаданные footer: ключ `c1.meta` с `source,symbol,tf,rows,data_hash,build_signature`.

## Конфиги (пример)

* `configs/datalayer.yaml`: API, store, ресемплинг, пороги C1.
* `configs/dataquality.yaml`: допуски/окна/пороги C2, пути вывода.
* `configs/features.yaml`: окна C3, путь фич.
* `configs/regime.yaml`: параметры ансамбля и детекторов C4.

## Контракты функций

* `align_and_flag_gaps(df_1m, strict=True) -> (df_aligned, ValidateStats)`
* `resample_1m_to(df_1m, dst_tf) -> df_tf`
* `validate(df, tf, symbol=None, repair=True, config=None, ref_1m=None) -> (df_clean, issues_df)`
* `build_features(df_in, symbol, tf, cfg=None) -> df_features`
* `infer_regime(df_in, tf, cfg=None) -> df_regime`

## Совместимость/миграция

* Старые импорты `c4_regime` перенаправлены в `ohlcv.regime`.
* CLI-совместимость: `report-missing`, `c1-update`, `c1-*`, `c2-validate` сохранены.

## Требования

* Python ≥ 3.10, `pandas>=2.0`, `numpy>=1.23`, `click>=8.1`, `requests>=2.28`, опционально `pyarrow>=14`.
