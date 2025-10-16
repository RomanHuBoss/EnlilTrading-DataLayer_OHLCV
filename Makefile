.PHONY: install dev test lint type fmt features example clean

PY ?= python
PIP ?= pip

install:
	$(PIP) install -U pip
	$(PIP) install -e .

dev:
	$(PIP) install -U pip
	$(PIP) install -e .[dev]

lint:
	ruff check .
	black --check .
	isort --check-only .

type:
	mypy ohlcv

fmt:
	black .
	isort .

test:
	$(PY) -m pytest -q --maxfail=1 --disable-warnings --cov=ohlcv --cov-report=term-missing

features:
	features-core build \
		--input data/ohlcv.csv \
		--symbol BTCUSDT \
		--tf 5m \
		--config configs/features.example.yaml \
		--output out/BTCUSDT_5m_features.parquet

example:
	@for TF in 1m 5m 15m 1h; do \
	  features-core build \
	    --input data/BTCUSDT_$$TF.csv \
	    --symbol BTCUSDT \
	    --tf $$TF \
	    --output out/BTCUSDT_$$TF.parquet ; \
	done

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info coverage.xml
