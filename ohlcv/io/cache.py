# Дисковый кэш исходных ответов для уменьшения повторных запросов.
import json
import hashlib
from pathlib import Path
from typing import Optional

def _key(symbol: str, start_iso: str, end_iso: str) -> str:
    raw = f"{symbol}|{start_iso}|{end_iso}"
    return hashlib.sha256(raw.encode()).hexdigest()

def cache_path(root: Path, symbol: str, start_iso: str, end_iso: str) -> Path:
    return root / symbol / f"{_key(symbol, start_iso, end_iso)}.json"

def get_cached(root: Path, symbol: str, start_iso: str, end_iso: str) -> Optional[list]:
    p = cache_path(root, symbol, start_iso, end_iso)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None

def put_cache(root: Path, symbol: str, start_iso: str, end_iso: str, rows: list) -> None:
    p = cache_path(root, symbol, start_iso, end_iso)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
