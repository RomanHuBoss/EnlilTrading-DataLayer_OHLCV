# HOWTO: запуск C4.Regime

## CLI

```bash
python -m c4_regime.cli infer \
  --features /data/features/BTCUSDT_5m.parquet \
  --context /data/context/BTCUSDT_5m.parquet \
  --config c4_regime/config/regime.default.yaml \
  --output /data/regime/BTCUSDT_5m.parquet
```

## Python API

```python
import pandas as pd
from c4_regime import infer_regime
from c4_regime.core import DEFAULT_CFG

feats = pd.read_parquet("/data/features/BTCUSDT_5m.parquet")
ctx = pd.read_parquet("/data/context/BTCUSDT_5m.parquet")  # опционально
out = infer_regime(feats, context=ctx, cfg=DEFAULT_CFG)
```

## Конфигурация

См. `c4_regime/config/regime.default.yaml`. Изменяйте пороги D1/D2, параметры D3 (lambda, stream), порог calm для D4, а также N_conf/N_cooldown.
