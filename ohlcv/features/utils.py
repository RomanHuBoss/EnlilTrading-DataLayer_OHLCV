from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class BuildMeta:
    component: str
    version: str
    params: Dict[str, Any]

    def build_version(self) -> str:
        # Try to embed a short git sha if available
        sha = ""
        try:
            import subprocess
            out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            sha = out.decode().strip()
        except Exception:
            sha = "nogit"
        payload = {"sha": sha, "params": self.params, "component": self.component, "version": self.version}
        h = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
        return f"{self.component}@{self.version}+{h}"
