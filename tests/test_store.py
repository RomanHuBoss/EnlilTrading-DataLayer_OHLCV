# Тест хранилища Parquet (структурные проверки).
import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timezone

from ohlcv.io.parquet_store import write_idempotent, parquet_path

def test_write_and_merge(tmp_path: Path):
    root = tmp_path
    symbol = "TESTUSDT"
    tf = "1m"
    idx1 = pd.date_range("2024-01-01T00:00:00Z", periods=3, freq="min")
    df1 = pd.DataFrame({"o":[1,2,3],"h":[1,2,3],"l":[1,2,3],"c":[1,2,3],"v":[1,1,1]}, index=idx1)
    p = write_idempotent(root, symbol, tf, df1)
    assert p.exists()

    idx2 = pd.date_range("2024-01-01T00:02:00Z", periods=3, freq="min")
    df2 = pd.DataFrame({"o":[9,9,9],"h":[9,9,9],"l":[9,9,9],"c":[9,9,9],"v":[2,2,2]}, index=idx2)
    p = write_idempotent(root, symbol, tf, df2)

    df = pd.read_parquet(p)
    # Должно быть 5 уникальных минут (0..4), последняя версия на пересечении
    assert len(df) == 5
    assert df.iloc[-1]["o"] == 9
