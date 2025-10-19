# ohlcv/version.py
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Any, Dict, Tuple

__version__ = "0.1.0"


# -------------------------
# Вспомогательные
# -------------------------

def _to_serializable(x: Any) -> Any:
    if x is None or isinstance(x, (int, float, str, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_to_serializable(i) for i in x]
    if isinstance(x, dict):
        return {str(k): _to_serializable(v) for k, v in x.items()}
    return str(x)


def _canonical_json(obj: Any) -> str:
    return json.dumps(_to_serializable(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _git_info() -> Tuple[str, bool]:
    """Возвращает (commit_sha, dirty). Пустая строка, False при недоступном git.
    Не бросает исключений.
    """
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return commit, dirty
    except Exception:
        return "", False


def _env_commit() -> str:
    for k in ("GIT_COMMIT", "CI_COMMIT_SHA", "SOURCE_VERSION", "BUILD_VCS_NUMBER"):
        v = os.getenv(k, "").strip()
        if v:
            return v
    return ""


# -------------------------
# Публичное API
# -------------------------

def build_signature(params: Dict[str, Any] | None = None, *, component: str = "C1") -> str:
    """Возвращает строку сигнатуры сборки: "C1-<ver>+p<psha>(+g<gsha>[-dirty])".

    psha — sha1 от канонического JSON параметров; gsha — короткий git SHA, если доступен.
    """
    params = params or {}
    psha = _sha1_hex(_canonical_json(params))[:8]
    gsha, dirty = _git_info()
    if not gsha:
        gsha = _env_commit()
        dirty = False

    sig = f"{component}-{__version__}+p{psha}"
    if gsha:
        sig += "+g" + gsha[:8]
        if dirty:
            sig += "-dirty"
    return sig
