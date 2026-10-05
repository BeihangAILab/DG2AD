"""Create a minimal environment and run the offline DGA2D smoke check."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parent
SUPPORTED_SMOKE_VERSIONS = {(3, 10), (3, 11), (3, 12), (3, 13)}
STAMP_NAME = ".dga2d-smoke-requirements.sha256"


def is_supported_version(version_info: Any) -> bool:
    return (int(version_info.major), int(version_info.minor)) in SUPPORTED_SMOKE_VERSIONS


def venv_python(venv_dir: Path, os_name: str = os.name) -> Path:
    if os_name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def requirements_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _interpreter_version(
    python_executable: Path,
    project_root: Path,
    runner: Callable[..., Any],
) -> tuple[int, int] | None:
    result = runner(
        [
            str(python_executable),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        major, minor = result.stdout.strip().split(".", maxsplit=1)
        return int(major), int(minor)
    except (AttributeError, TypeError, ValueError):
        return None


def run_quickstart(
    project_root: Path = PROJECT_ROOT,
    base_python: Path | None = None,
    version_info: Any = sys.version_info,
    os_name: str = os.name,
    runner: Callable[..., Any] = subprocess.run,
) -> int:
    project_root = project_root.resolve()
    base_python = Path(base_python or sys.executable).resolve()
    if not is_supported_version(version_info):
        print(
            "[ERROR] DGA2D smoke requires standard CPython 3.10 through 3.13. ",
            file=sys.stderr,
            flush=True,
        )
        return 2

    requirements = project_root / "requirements-smoke.txt"
    main_script = project_root / "main.py"
    if not requirements.is_file() or not main_script.is_file():
        print(
            "[ERROR] Run quickstart.py from a complete DGA2D checkout.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    venv_dir = project_root / ".venv"
    python_executable = venv_python(venv_dir, os_name)
    if venv_dir.exists() and not python_executable.is_file():
        print(
            f"[ERROR] Existing virtual environment is incomplete: {venv_dir}\n"
            "Move or remove it, then run quickstart.py again.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    if not venv_dir.exists():
        print(
            f"[SETUP] Creating {venv_dir.name} with Python "
            f"{version_info.major}.{version_info.minor}",
            flush=True,
        )
        result = runner(
            [str(base_python), "-m", "venv", str(venv_dir)],
            cwd=project_root,
        )
        if result.returncode != 0:
            return int(result.returncode)
        if not python_executable.is_file():
            print(
                "[ERROR] Virtual environment creation did not produce Python.",
                file=sys.stderr,
                flush=True,
            )
            return 2

    interpreter_version = _interpreter_version(python_executable, project_root, runner)
    if interpreter_version not in SUPPORTED_SMOKE_VERSIONS:
        shown = (
            "unknown"
            if interpreter_version is None
            else f"{interpreter_version[0]}.{interpreter_version[1]}"
        )
        print(
            f"[ERROR] {venv_dir.name} uses unsupported Python {shown}. "
            "Move or remove it, then run quickstart.py with standard CPython 3.10-3.13.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    digest = requirements_digest(requirements)
    stamp = venv_dir / STAMP_NAME
    installed_digest = stamp.read_text(encoding="ascii").strip() if stamp.is_file() else ""
    if installed_digest != digest:
        print("[SETUP] Installing minimal smoke dependencies", flush=True)
        result = runner(
            [
                str(python_executable),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--quiet",
                "-r",
                str(requirements),
            ],
            cwd=project_root,
        )
        if result.returncode != 0:
            return int(result.returncode)
        stamp.write_text(digest + "\n", encoding="ascii")
    else:
        print("[SETUP] Minimal dependencies are up to date", flush=True)

    print("[RUN] Starting the offline FSSP smoke check", flush=True)
    result = runner(
        [str(python_executable), str(main_script), "+experiment=smoke"],
        cwd=project_root,
    )
    return int(result.returncode)


def main() -> int:
    return run_quickstart()


if __name__ == "__main__":
    raise SystemExit(main())
