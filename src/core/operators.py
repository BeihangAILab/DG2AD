from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import math
import multiprocessing
import os
import re
import shutil
import time
import sys
import traceback
from pathlib import Path
from typing import Iterable

from .execution import SolutionState


VERSION_PATTERN = re.compile(r"v(\d+)\.py$")


def discover_operator_versions(
    slots_dir: str | Path, allowed_nodes: set[str] | None = None
) -> dict[str, list[str]]:
    slots = Path(slots_dir)
    discovered: dict[str, list[str]] = {}
    if not slots.is_dir():
        return discovered
    for category in sorted(path for path in slots.iterdir() if path.is_dir()):
        if category.name == "_archive":
            continue
        for node_dir in sorted(path for path in category.iterdir() if path.is_dir()):
            key = f"{category.name}|{node_dir.name}"
            if allowed_nodes is not None and key not in allowed_nodes:
                continue
            versions = []
            for file_path in node_dir.glob("v*.py"):
                match = VERSION_PATTERN.fullmatch(file_path.name)
                if match:
                    versions.append((int(match.group(1)), f"v{match.group(1)}"))
            if versions:
                discovered[key] = [version for _, version in sorted(versions)]
    if allowed_nodes is not None:
        missing = sorted(set(allowed_nodes) - set(discovered))
        if missing:
            raise FileNotFoundError(
                "Allowed operator nodes have no executable version files: "
                + ", ".join(missing)
            )
    return discovered


def _retry_windows_file_operation(operation, attempts: int = 4) -> None:
    """Retry short-lived Windows file locks without hiding persistent failures."""
    for attempt in range(attempts):
        try:
            operation()
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05 * (attempt + 1))


class RunOperatorPool:
    """Isolated, recoverable implementation pool for one training run."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.archive_dir = self.root / "_archive"

    @classmethod
    def create_from_source(
        cls,
        source_slots: str | Path,
        destination: str | Path,
        selected_nodes: Iterable[str],
    ) -> "RunOperatorPool":
        source = Path(source_slots).resolve()
        destination = Path(destination).resolve()
        selected_nodes = list(selected_nodes)
        destination.mkdir(parents=True, exist_ok=True)
        for node in sorted(set(selected_nodes)):
            category, name = node.split("|", 1)
            source_dir = source / category / name
            if not source_dir.is_dir():
                raise FileNotFoundError(f"Baseline operator directory not found: {source_dir}")
            target_dir = destination / category / name
            target_dir.mkdir(parents=True, exist_ok=True)
            for version_path in source_dir.glob("v*.py"):
                if VERSION_PATTERN.fullmatch(version_path.name):
                    shutil.copy2(version_path, target_dir / version_path.name)
        pool = cls(destination)
        missing = [node for node in selected_nodes if not pool.available_versions(node)]
        if missing:
            raise ValueError(f"Selected nodes have no copied implementations: {missing}")
        return pool

    def node_dir(self, component: str) -> Path:
        category, name = component.split("|", 1)
        return self.root / category / name

    def version_path(self, component: str, version: str) -> Path:
        return self.node_dir(component) / f"{version}.py"

    def available_versions(self, component: str) -> list[str]:
        versions = []
        directory = self.node_dir(component)
        if not directory.is_dir():
            return []
        for path in directory.glob("v*.py"):
            match = VERSION_PATTERN.fullmatch(path.name)
            if match:
                versions.append((int(match.group(1)), f"v{match.group(1)}"))
        return [version for _, version in sorted(versions)]

    def read_code(self, component: str, version: str) -> str:
        return self.version_path(component, version).read_text(encoding="utf-8")

    def next_version(self, component: str) -> str:
        values = [int(version[1:]) for version in self.available_versions(component)]
        category, name = component.split("|", 1)
        archived = self.archive_dir / category / name
        if archived.is_dir():
            for path in archived.glob("v*.py"):
                match = re.match(r"v(\d+)(?:__|\.py)", path.name)
                if match:
                    values.append(int(match.group(1)))
        return f"v{max(values, default=0) + 1}"

    def add_code(self, component: str, code: str) -> str:
        version = self.next_version(component)
        target = self.version_path(component, version)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".py.tmp")
        temporary.write_text(code, encoding="utf-8")
        os.replace(temporary, target)
        return version

    def archive(self, component: str, version: str, reason: str) -> Path:
        active = self.version_path(component, version)
        if not active.exists():
            raise FileNotFoundError(active)
        if len(self.available_versions(component)) <= 1:
            raise ValueError(f"Cannot remove the only implementation of {component}")
        category, name = component.split("|", 1)
        destination_dir = self.archive_dir / category / name
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{version}__{reason}.py"
        suffix = 1
        while destination.exists():
            destination = destination_dir / f"{version}__{reason}_{suffix}.py"
            suffix += 1
        shutil.move(str(active), str(destination))
        return destination

    def replace(self, component: str, old_version: str, new_code: str) -> str:
        new_version = self.add_code(component, new_code)
        try:
            self.archive(component, old_version, "replaced")
        except Exception:
            self.version_path(component, new_version).unlink(missing_ok=True)
            raise
        return new_version

    def restore_snapshot(self, snapshot_dir: str | Path) -> None:
        snapshot = Path(snapshot_dir).resolve()
        if not snapshot.is_dir():
            raise FileNotFoundError(snapshot)
        backup = self.root.with_name(self.root.name + ".restore_backup")
        if backup.exists():
            _retry_windows_file_operation(lambda: shutil.rmtree(backup))
        if self.root.exists():
            _retry_windows_file_operation(lambda: os.replace(self.root, backup))
        try:
            shutil.copytree(snapshot, self.root)
            if backup.exists():
                _retry_windows_file_operation(lambda: shutil.rmtree(backup))
        except Exception:
            if self.root.exists():
                _retry_windows_file_operation(lambda: shutil.rmtree(self.root))
            if backup.exists():
                _retry_windows_file_operation(lambda: os.replace(backup, self.root))
            raise

    def file_hashes(self) -> dict[str, str]:
        hashes = {}
        for path in sorted(self.root.rglob("*.py")):
            hashes[str(path.relative_to(self.root)).replace("\\", "/")] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        return hashes


def validate_candidate_code(code: str):
    """Validate the syntax and exact public contract of generated operator code."""
    try:
        tree = ast.parse(code)
        compile(tree, "<generated-operator>", "exec")
    except (SyntaxError, ValueError) as exc:
        return False, f"Generated code does not compile: {exc}"
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
    ]
    if len(definitions) != 1 or isinstance(definitions[0], ast.AsyncFunctionDef):
        return False, "Generated code must define exactly one synchronous run function"
    args = definitions[0].args
    positional = list(args.posonlyargs) + list(args.args)
    if (
        [arg.arg for arg in positional] != ["env_data", "state", "calc_makespan_fn"]
        or args.vararg is not None
        or args.kwarg is not None
        or args.kwonlyargs
    ):
        return False, ("run signature must be exactly run(env_data, state, calc_makespan_fn)")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "calc_makespan_fn":
            first = node.args[0]
            if isinstance(first, ast.Name) and first.id == "env_data":
                return False, (
                    "calc_makespan_fn arguments are reversed; the candidate/state "
                    "must be the first argument"
                )
    return True, None


def _runtime_validation_worker(code, category, env_data, evaluator_module_name, problem_dir, result_pipe):
    try:
        if problem_dir and problem_dir not in sys.path:
            sys.path.insert(0, problem_dir)
        evaluator = importlib.import_module(evaluator_module_name)
        namespace = {"__name__": "generated_operator_runtime_validation"}
        exec(compile(code, "<generated-operator-runtime_validation>", "exec"), namespace)
        state = SolutionState()
        if not category.startswith("initialization"):
            evaluator.initialize_state(env_data, state)
            if state.sequence is None:
                raise ValueError("Domain initialisation produced no sequence")
            state.metadata["prev_sequence"] = copy.deepcopy(state.sequence)
            state.metadata["prev_makespan"] = float(state.makespan)

        def calculate(candidate, _candidate_env=None):
            return evaluator.calc_makespan(candidate, env_data)

        state = namespace["run"](env_data, state, calculate)
        if state is None or not hasattr(state, "sequence"):
            raise TypeError("Generated operator did not return a SolutionState")
        actual = float(evaluator.calc_makespan(state.sequence, env_data))
        invalid = getattr(evaluator, "INVALID_SCORE", None)
        if not math.isfinite(actual):
            raise ValueError("Generated operator produced a non-finite objective")
        if invalid is not None and actual >= float(invalid):
            raise ValueError(f"Generated operator produced an invalid solution (score={actual})")
        result_pipe.send((True, None))
    except BaseException:
        result_pipe.send((False, traceback.format_exc(limit=8)))
    finally:
        result_pipe.close()


def validate_candidate_runtime(
    code,
    category,
    env_data,
    evaluator_module_name,
    problem_dir,
    timeout_seconds=30.0,
):
    """Validate generated code in an isolated spawned process."""
    if env_data is None or not evaluator_module_name:
        return True, None
    context = multiprocessing.get_context("spawn")
    receive_pipe, send_pipe = context.Pipe(duplex=False)
    process = context.Process(
        target=_runtime_validation_worker,
        args=(
            code,
            category,
            env_data,
            evaluator_module_name,
            problem_dir,
            send_pipe,
        ),
    )
    process.start()
    send_pipe.close()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        receive_pipe.close()
        return False, f"Generated operator runtime validation exceeded {timeout_seconds:.1f}s"
    if receive_pipe.poll(1.0):
        result = receive_pipe.recv()
        receive_pipe.close()
        return result
    receive_pipe.close()
    return False, (
        "Generated operator runtime validation process exited without returning a result "
        f"(exit code {process.exitcode})"
    )
