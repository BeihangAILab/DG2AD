from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

from .credit import TransitionCreditStore
from .operators import RunOperatorPool
from .structure import DirectedOperatorGraph

SCHEMA_NAME = "dga2d-training"
SCHEMA_VERSION = 3


def _retry_windows_file_operation(operation, attempts: int = 4) -> None:
    """Retry short-lived Windows file locks without weakening rollback."""
    for attempt in range(attempts):
        try:
            operation()
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05 * (attempt + 1))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def capture_rng_state(local_rng: random.Random, torch_module) -> dict[str, Any]:
    state: dict[str, Any] = {
        "local_random": local_rng.getstate(),
        "python_random": random.getstate(),
        "torch_cpu": torch_module.get_rng_state(),
    }
    if torch_module.cuda.is_available():
        state["torch_cuda"] = torch_module.cuda.get_rng_state_all()
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    return state


def restore_rng_state(state: dict[str, Any], local_rng: random.Random, torch_module):
    local_rng.setstate(state["local_random"])
    random.setstate(state["python_random"])
    torch_module.set_rng_state(state["torch_cpu"])
    if torch_module.cuda.is_available() and "torch_cuda" in state:
        torch_module.cuda.set_rng_state_all(state["torch_cuda"])
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except Exception:
            pass


class CheckpointManager:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).resolve()
        self.root = self.run_dir / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, kind: str) -> Path:
        if kind not in {"best", "last"}:
            raise ValueError(f"Unknown checkpoint kind: {kind}")
        return self.root / kind

    def exists(self, kind: str) -> bool:
        return (self.path(kind) / "checkpoint.json").is_file()

    def clear(self) -> None:
        for kind in ("best", "last"):
            directory = self.path(kind)
            if directory.exists():
                shutil.rmtree(directory)

    def save(
        self,
        kind: str,
        generation: int,
        graph: DirectedOperatorGraph,
        credit: TransitionCreditStore,
        policy,
        pool: RunOperatorPool,
        manifest: dict[str, Any],
        metadata: dict[str, Any],
        llm_state: dict[str, Any],
        local_rng: random.Random,
    ) -> Path:
        target = self.path(kind)
        staging = self.root / f".{kind}.staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        _write_json(staging / "graph.json", graph.to_dict())
        _write_json(staging / "markov_q.json", credit.to_dict())
        _write_json(staging / "manifest.json", manifest)
        _write_json(staging / "aliases.json", policy.alias_by_action)
        _write_json(
            staging / "checkpoint.json",
            {
                "schema": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "kind": kind,
                "generation": int(generation),
                "structure": manifest.get("structure", {}).get("name"),
                "metadata": metadata,
                "llm": llm_state,
            },
        )
        shutil.copytree(
            pool.root,
            staging / "operator_pool",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        policy.save_adapter(staging)
        state = {
            "policy": policy.training_state(),
            "rng": capture_rng_state(local_rng, policy.torch),
        }
        policy.torch.save(state, staging / "training_state.pt")

        backup = self.root / f".{kind}.previous"
        if backup.exists():
            _retry_windows_file_operation(lambda: shutil.rmtree(backup))
        if target.exists():
            _retry_windows_file_operation(lambda: os.replace(target, backup))
        try:
            _retry_windows_file_operation(lambda: os.replace(staging, target))
            if backup.exists():
                _retry_windows_file_operation(lambda: shutil.rmtree(backup))
        except Exception:
            if target.exists():
                _retry_windows_file_operation(lambda: shutil.rmtree(target))
            if backup.exists():
                _retry_windows_file_operation(lambda: os.replace(backup, target))
            raise
        return target

    def load(
        self,
        kind: str,
        policy,
        pool: RunOperatorPool,
        local_rng: random.Random,
        load_optimizer: bool,
    ) -> dict[str, Any]:
        directory = self.path(kind)
        checkpoint = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
        if (
            checkpoint.get("schema") != SCHEMA_NAME
            or int(checkpoint.get("schema_version", -1)) != SCHEMA_VERSION
        ):
            raise ValueError(
                f"Checkpoint schema is not compatible with {SCHEMA_NAME} v{SCHEMA_VERSION}"
            )
        try:
            state = policy.torch.load(
                directory / "training_state.pt", map_location="cpu", weights_only=False
            )
        except TypeError:
            state = policy.torch.load(directory / "training_state.pt", map_location="cpu")
        policy.load_training_state(state["policy"], load_optimizer=load_optimizer)
        restore_rng_state(state["rng"], local_rng, policy.torch)
        pool.restore_snapshot(directory / "operator_pool")
        graph = DirectedOperatorGraph.from_dict(
            json.loads((directory / "graph.json").read_text(encoding="utf-8"))
        )
        credit = TransitionCreditStore.from_dict(
            json.loads((directory / "markov_q.json").read_text(encoding="utf-8"))
        )
        policy.set_graph(graph)
        return {
            "checkpoint": checkpoint,
            "graph": graph,
            "credit": credit,
            "manifest": json.loads((directory / "manifest.json").read_text(encoding="utf-8")),
        }
