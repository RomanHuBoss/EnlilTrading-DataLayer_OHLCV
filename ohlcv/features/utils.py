from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class BuildMeta:
    component: str
    version: str
    params: Dict[str, Any]

    def build_version(self) -> str:
        """Версия сборки признаков на базе git-ревизии и параметров окон."""
        sha = "nogit"
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            sha = out.decode().strip()
        except Exception:
            pass
        payload = {
            "sha": sha,
            "params": self.params,
            "component": self.component,
            "version": self.version,
        }
        h = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        return f"{self.component}@{self.version}+{h}"
