# HOWTO: C1.DataLayer.OHLCV

## 1) Получи архив
Скачай репозиторий в формате zip-архива

## 2) Разверни проект локально
```bash
unzip DataLayer_OHLCV.zip -d DataLayer_OHLCV
cd DataLayer_OHLCV
```

## 3) Подготовь окружение
```bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e .
```

## 4) Задай переменные
```bash
cp .env.example .env
# заполни BYBIT_API_KEY и BYBIT_API_SECRET при наличии
export $(grep -v '^#' .env | xargs)
mkdir -p "${DATA_ROOT:-/data}"
```

## 5) Прогони юнит-тесты
```bash
pip install pytest
pytest -q
```

## 6) Догрузи историю 1m
```bash
python -m ohlcv.cli backfill   --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT   --since 2024-10-15
```

## 7) Обновляй хвост (≤ 1 бар задержки)
```bash
python -m ohlcv.cli update --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT
```

## 8) Ресемплинг в 5m
```bash
python -m ohlcv.cli resample --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT --from-tf 1m --to-tf 5m
```

## 9) Ресемплинг в 15m
```bash
python -m ohlcv.cli resample --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT --from-tf 1m --to-tf 15m
```

## 10) Ресемплинг в 1h
```bash
python -m ohlcv.cli resample --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT --from-tf 1m --to-tf 1h
```

## 11) Отчёт о пропусках (DoD)
```bash
python -m ohlcv.cli report-missing --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT --tf 1m --out missing_1m.csv
```

## 12) Проверка файлов данных
Структура после выполнения:
```
${DATA_ROOT:-/data}/
  BTCUSDT/
    1m.parquet
    5m.parquet
    15m.parquet
    1h.parquet
  ETHUSDT/
    ...
```

### Примечание по чтению Parquet
```python
import pandas as pd
df = pd.read_parquet("/data/BTCUSDT/1m.parquet")
```

### Эксплуатация по расписанию
- Обновление хвоста (п.7) каждую минуту.
- Ресемплинг (пп.8–10) после обновления 1m.
- Отчёт о пропусках (п.11) ежедневно.
