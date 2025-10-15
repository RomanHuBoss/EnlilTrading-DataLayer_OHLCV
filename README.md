# C1.DataLayer.OHLCV

Версия: 1.0
Дата генерации: 2025-10-15T18:51:50.724298Z

Назначение: надёжная загрузка, ресемплинг и хранение OHLCV‑рядов. Базовый ТФ 1m, производные ТФ 5m/15m/1h.
Хранение в Parquet с метаданными, кэшем и идемпотентным догрузом истории. Все времена — UTC.

Входы: API Bybit (кастодиальные ключи read‑only), список тикеров.
Выходы: `/data/{symbol}/{tf}.parquet` со схемой: ts, o, h, l, c, v (+ опционально t).

NFR:
- Пропуски < 0.01%
- Валидность таймстампов (минутные границы, монотонность)
- Задержка обновления ≤ 1 бар

DoD:
- Юнит‑тесты: ресемплинг/валидация
- Интеграционная догрузка ≥1 год по ≥5 тикерам
- Отчёт о пропусках (CSV)

Запуск (пример):
```
python -m ohlcv.cli backfill --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT --since 2024-10-15 --tf 1m
python -m ohlcv.cli update --symbols BTCUSDT,ETHUSDT --tf 1m
python -m ohlcv.cli resample --symbols BTCUSDT,ETHUSDT --from-tf 1m --to-tf 5m
python -m ohlcv.cli report-missing --symbols BTCUSDT,ETHUSDT --tf 1m --out /mnt/data/missing_report.csv
```
