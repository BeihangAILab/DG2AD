from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def cfg_get(config: Any, path: str, default: Any = None) -> Any:
    """Read a dotted path from DictConfig, dict, or simple namespace objects."""

    current = config
    for part in path.split("."):
        if current is None:
            return default
        if isinstance(current, Mapping):
            if part not in current:
                return default
            current = current[part]
            continue
        try:
            current = getattr(current, part)
        except (AttributeError, TypeError):
            return default
    return current


def resolve_project_path(project_root: str | Path, configured: str | Path) -> Path:
    path = Path(configured)
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve()


def configure_dataset_root(project_root: str | Path, configured: str | Path) -> Path:
    """Expose the resolved benchmark root to domain evaluator subprocesses."""

    root = resolve_project_path(project_root, configured)
    os.environ["DGA2D_DATA_ROOT"] = str(root)
    return root


def problem_data_dir(problem: str, private_fallback_directory: str | Path) -> Path:
    """Resolve one domain's data directory with a private-repo fallback.

    Third-party benchmark files are stored outside ``src``.
    The fallback keeps existing private checkouts usable while data is migrated
    to ``dataset.root``.
    """

    private_fallback = Path(private_fallback_directory).resolve()
    configured = os.environ.get("DGA2D_DATA_ROOT", "").strip()
    if not configured:
        return private_fallback
    candidate = (Path(configured) / problem).resolve()
    if candidate.exists() or not private_fallback.exists():
        return candidate
    return private_fallback


def configure_huggingface_cache(project_root: str | Path, configured: str | Path) -> Path:
    root = resolve_project_path(project_root, configured)
    root.mkdir(parents=True, exist_ok=True)
    values = {
        "HF_HOME": root,
        "HF_HUB_CACHE": root / "hub",
        "HF_XET_CACHE": root / "xet",
        "TRANSFORMERS_CACHE": root / "transformers",
        "TMP": root / "tmp",
        "TEMP": root / "tmp",
        "TMPDIR": root / "tmp",
    }
    for name, value in values.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(value)
    return root
