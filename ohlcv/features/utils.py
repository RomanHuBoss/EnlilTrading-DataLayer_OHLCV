# ohlcv/features/utils.py
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

# Канонический JSON без пробелов, стабильная сортировка — для детерминированного хэша
_DEF_JSON_KW = dict(separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def _to_serializable(x: Any) -> Any:
    if x is None or isinstance(x, (int, float, str, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_to_serializable(i) for i in x]
    if isinstance(x, dict):
        return {str(k): _to_serializable(v) for k, v in x.items()}
    # Fallback: строковое представление
    return str(x)


def canonical_json(obj: Any) -> str:
    return json.dumps(_to_serializable(obj), **_DEF_JSON_KW)


def sha1_hex(data: str) -> str:
    return hashlib.sha1(data.encode("utf-8")).hexdigest()


def _probe_git() -> Tuple[str, bool]:
    """Возвращает (commit_sha, is_dirty). Пустая строка, False при недоступном git.

    Не бросает исключений.
    """
    try:
        # Проверка, что мы в репозитории
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        # Наличие незакоммиченных изменений
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        dirty = bool(status.strip())
        return commit, dirty
    except Exception:
        return "", False


def _env_git_fallback() -> str:
    # Популярные переменные CI
    for key in ("GIT_COMMIT", "CI_COMMIT_SHA", "BUILD_VCS_NUMBER", "SOURCE_VERSION"):
        v = os.getenv(key, "").strip()
        if v:
            return v
    return ""


@dataclass
class BuildMeta:
    component: str
    version: str
    params: Dict[str, Any] = field(default_factory=dict)

    def params_hash(self) -> str:
        cj = canonical_json(self.params)
        return sha1_hex(cj)

    def git_commit(self) -> Tuple[str, bool]:
        commit, dirty = _probe_git()
        if not commit:
            env_commit = _env_git_fallback()
            if env_commit:
                return env_commit, False
        return commit, dirty

    def build_version(self) -> str:
        psha = self.params_hash()[:8]
        gsha, dirty = self.git_commit()
        if gsha:
            gsha = gsha[:8] + ("-dirty" if dirty else "")
        tag = f"{self.component}-{self.version}+p{psha}"
        if gsha:
            tag += f"+g{gsha}"
        return tag
