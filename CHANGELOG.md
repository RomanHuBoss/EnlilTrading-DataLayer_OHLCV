# Changelog

## [0.3.0] — C3 интеграция

* Добавлен подпакет `ohlcv.features` (core/cli/schema/utils).
* CLI `features-core build` с поддержкой CSV/Parquet и флага `--strict`.
* Нормализация и строгая валидация входной схемы (`ohlcv/features/schema.py`).
* Набор признаков по умолчанию и инварианты (см. `docs/C3_FEATURES.md`).
* Тесты: `tests/test_features_strict.py`, `tests/test_features_cli_integration.py`.
* CI: smoke для `features-core`; матрица Python 3.11/3.12; линтинг/типизация.
* Зависимости: `pandas>=2.2`, `pyyaml>=6`.

## [0.2.x]

* C2 · DataQuality: улучшения в отчётах `issues` и `quality_summary`.
* C1 · DataLayer: стабильность backfill/update, отчёты пропусков.

## [0.2.0]

* Стандартизация формата Parquet и метаданных `c1.meta`.

## [0.1.0]

* Инициализация репозитория: C1/C2.
