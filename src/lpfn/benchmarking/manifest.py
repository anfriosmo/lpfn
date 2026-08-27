from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def source_tree_sha256(root: Path) -> str:
    """Hash Python sources and benchmark scripts in deterministic path order."""
    h = hashlib.sha256()
    files = sorted((root / "src").rglob("*.py")) + sorted((root / "benchmarks").rglob("*.py"))
    for path in files:
        rel = path.relative_to(root).as_posix().encode("utf-8")
        h.update(rel)
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def build_manifest(*, root: Path, config: dict[str, object], command: list[str] | None = None) -> dict[str, object]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "default_dtype": str(torch.get_default_dtype()),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        },
        "source": {
            "git_commit": _git_commit(root),
            "tree_sha256": source_tree_sha256(root),
        },
        "command": command or [],
    }


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
