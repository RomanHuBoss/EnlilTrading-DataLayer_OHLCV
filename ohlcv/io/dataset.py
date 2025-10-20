# ohlcv/io/dataset.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .parquet_store import parquet_path, write_idempotent
from ..core.resample import resample_1m_to
from ..quality.validator import validate, QualityConfig
from ..quality.issues import summarize_issues

__all__ = [
    "read",
    "read_range",
    "read_meta",
    "write",
    "sanitize",
    "sanitize_and_write",
    "resample_and_write",
    "list_symbols",
    "list_tfs",
    "manifest",
]

# ---------------- Internals ----------------

_DEF_ORDER: List[str] = ["ts", "o", "h", "l", "c", "v", "t", "is_gap"]


def _empty_df(columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    cols = [c for c in (columns or _DEF_ORDER) if c in _DEF_ORDER]
    if "ts" not in cols:
        cols = ["ts"] + cols
    dtypes: Dict[str, Any] = {
        "ts": "int64",
        "o": "float64",
        "h": "float64",
        "l": "float64",
        "c": "float64",
        "v": "float64",
        "t": "float64",
        "is_gap": "bool",
    }
    df = pd.DataFrame({c: pd.Series(dtype=dtypes.get(c, "float64")) for c in cols})
    # enforce exact dtype
    for c, dt in dtypes.items():
        if c in df.columns:
            df[c] = df[c].astype(dt)
    return df[cols]


def _canonicalize(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    out = df.copy()
    # normalize column names to lower-case canonical ones if user passed different register
    rename = {c: str(c).lower() for c in out.columns}
    out = out.rename(columns=rename)
    # ensure types
    if "ts" in out.columns:
        out["ts"] = pd.to_numeric(out["ts"], errors="coerce").astype("Int64").fillna(0).astype("int64")
    for c in ("o", "h", "l", "c", "v", "t"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    if "is_gap" in out.columns:
        out["is_gap"] = out["is_gap"].astype(bool)
    # sort + dedup by ts (last wins) to satisfy C1 invariants on read
    if "ts" in out.columns:
        out = out.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    # order
    want = [c for c in _DEF_ORDER if c in out.columns]
    if columns:
        cols = ["ts"] + [c for c in columns if c != "ts" and c in want]
        want = [c for c in want if c in cols]
    if "ts" in want:
        out = out[want]
    return out


# ---------------- Public API ----------------

def read(root: Path | str, symbol: str, tf: str, *, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Чтение датасета OHLCV из Parquet (C1). Возвращает DataFrame в канонической схеме и типах.
    """
    path = parquet_path(root, symbol, tf)
    if not Path(path).exists():
        return _empty_df(columns)
    # columns hint is best-effort; engine may ignore
    try:
        df = pd.read_parquet(path, columns=None if columns is None else list(set(["ts", *columns])))
    except Exception:
        df = pd.read_parquet(path)
    return _canonicalize(df, columns)


def read_range(
    root: Path | str,
    symbol: str,
    tf: str,
    *,
    ts_from: Optional[int] = None,
    ts_to: Optional[int] = None,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Чтение диапазона по ts (включительно) в канонической схеме.
    """
    cols = None if columns is None else list(set(["ts", *columns]))
    df = read(root, symbol, tf, columns=cols)
    if df.empty:
        return df
    m = pd.Series(True, index=df.index)
    if ts_from is not None:
        m &= df["ts"] >= int(ts_from)
    if ts_to is not None:
        m &= df["ts"] <= int(ts_to)
    out = df.loc[m]
    return out.reset_index(drop=True)


def read_meta(root: Path | str, symbol: str, tf: str) -> Dict[str, Any]:
    """
    Чтение footer-метаданных C1 (ключ b"c1.meta"). Если отсутствует — попытка sidecar .meta.json.
    """
    path = parquet_path(root, symbol, tf)
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import pyarrow.parquet as pq  # type: ignore
        md = pq.ParquetFile(str(p)).metadata.metadata
        if md and b"c1.meta" in md:
            return json.loads(md[b"c1.meta"].decode("utf-8"))
    except Exception:
        pass
    # sidecar
    sidecar = p.with_suffix(".meta.json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def write(
    root: Path | str,
    symbol: str,
    tf: str,
    df_new: pd.DataFrame,
    *,
    source: str = "unknown",
    compression: str | None = "zstd",
    zstd_level: int = 7,
    row_group_size: int = 256_000,
) -> Path:
    """
    Идемпотентная запись блока данных (см. C1). Делегирует write_idempotent.
    """
    return write_idempotent(
        root=root,
        symbol=symbol,
        tf=tf,
        df_new=df_new,
        compression=compression,
        zstd_level=zstd_level,
        row_group_size=row_group_size,
        source=source,
    )


def sanitize(
    df: pd.DataFrame,
    *,
    tf: str,
    symbol: Optional[str] = None,
    repair: bool = True,
    config: Optional[QualityConfig] = None,
    ref_1m: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Санитайзинг по C2. Возвращает: (df_clean, issues_df, issues_summary).
    """
    clean, issues = validate(df, tf=tf, symbol=symbol, repair=repair, config=config, ref_1m=ref_1m)
    return clean, issues, summarize_issues(issues)


def sanitize_and_write(
    root: Path | str,
    symbol: str,
    tf: str,
    df_new: pd.DataFrame,
    *,
    repair: bool = True,
    config: Optional[QualityConfig] = None,
    ref_1m: Optional[pd.DataFrame] = None,
    source: str = "unknown",
    compression: str | None = "zstd",
) -> Tuple[Path, pd.DataFrame]:
    """
    Санитайзинг C2 и запись по C1. Возвращает (path, issues_df).
    """
    clean, issues, _ = sanitize(df_new, tf=tf, symbol=symbol, repair=repair, config=config, ref_1m=ref_1m)
    path = write(
        root=root,
        symbol=symbol,
        tf=tf,
        df_new=clean,
        source=source,
        compression=compression,
    )
    return path, issues


def resample_and_write(
    root: Path | str,
    symbol: str,
    src_tf: str,
    dst_tf: str,
    *,
    source: str = "resample",
    compression: str | None = "zstd",
) -> Path:
    """
    Ресемплинг из src_tf в dst_tf по контракту C1/C2 и запись в хранилище.
    Для src_tf='1m' используется детерминированный ресемплер.
    """
    df_src = read(root, symbol, src_tf)
    if df_src.empty:
        return parquet_path(root, symbol, dst_tf)
    if src_tf != "1m":
        raise ValueError("ресемплинг поддержан только из '1m'")
    df_dst = resample_1m_to(df_src, dst_tf)
    return write(root, symbol, dst_tf, df_dst, source=source, compression=compression)


def list_symbols(root: Path | str) -> List[str]:
    """
    Список доступных символов (каталоги верхнего уровня, содержащие parquet).
    """
    res: List[str] = []
    for p in Path(root).glob("*"):
        if p.is_dir() and any((p.glob("*.parquet"))):
            res.append(p.name)
    return sorted(res)


def list_tfs(root: Path | str, symbol: str) -> List[str]:
    """
    Список доступных таймфреймов для символа.
    """
    base = Path(root) / str(symbol)
    if not base.exists():
        return []
    out = []
    for q in base.glob("*.parquet"):
        name = q.name
        if name.endswith(".parquet"):
            out.append(name[:-8])  # strip ".parquet"
    return sorted(out)


def manifest(root: Path | str, symbol: str, tf: str) -> Dict[str, Any]:
    """
    Краткое описание датасета: размеры, диапазон ts, хэш по C1, метаданные.
    """
    meta = read_meta(root, symbol, tf)
    p = parquet_path(root, symbol, tf)
    exists = Path(p).exists()
    rows = 0
    min_ts = 0
    max_ts = 0
    if exists:
        try:
            # читаем только ts для быстроты
            ts = pd.read_parquet(p, columns=["ts"])["ts"].astype("int64")
            rows = int(len(ts))
            if rows:
                min_ts = int(ts.iloc[0])
                max_ts = int(ts.iloc[-1])
        except Exception:
            pass
    return {
        "path": str(p),
        "exists": exists,
        "rows": rows,
        "min_ts": meta.get("min_ts", min_ts),
        "max_ts": meta.get("max_ts", max_ts),
        "data_hash": meta.get("data_hash", ""),
        "meta": meta,
    }
