from __future__ import annotations

import sys
from pathlib import Path

# Добавить каталог с пакетом "ohlcv" (родитель tests) в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
