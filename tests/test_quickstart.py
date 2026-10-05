import contextlib
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import quickstart


VERSION_310 = SimpleNamespace(major=3, minor=10)


class FakeRunner:
    def __init__(self, root: Path, os_name: str = "nt", failures=None):
        self.root = root
        self.os_name = os_name
        self.failures = failures or {}
        self.commands = []

    def __call__(self, command, **kwargs):
        command = [str(item) for item in command]
        self.commands.append(command)
        if command[1:3] == ["-m", "venv"]:
            return_code = self.failures.get("venv", 0)
            if return_code == 0:
                python_path = quickstart.venv_python(self.root / ".venv", self.os_name)
                python_path.parent.mkdir(parents=True, exist_ok=True)
                python_path.touch()
            return subprocess.CompletedProcess(command, return_code)
        if len(command) > 1 and command[1] == "-c":
            return subprocess.CompletedProcess(
                command,
                self.failures.get("version", 0),
                stdout="3.10\n",
                stderr="",
            )
        if "pip" in command:
            return subprocess.CompletedProcess(command, self.failures.get("pip", 0))
        if command[-1] == "+experiment=smoke":
            return subprocess.CompletedProcess(command, self.failures.get("smoke", 0))
        raise AssertionError(f"Unexpected command: {command}")


def make_checkout(root: Path):
    (root / "requirements-smoke.txt").write_text("numpy==1.26.4\n", encoding="utf-8")
    (root / "main.py").write_text("raise SystemExit(0)\n", encoding="utf-8")


class QuickstartTests(unittest.TestCase):
    def test_platform_specific_venv_python_paths(self):
        root = Path("checkout") / ".venv"
        self.assertEqual(
            quickstart.venv_python(root, "nt"),
            root / "Scripts" / "python.exe",
        )
        self.assertEqual(
            quickstart.venv_python(root, "posix"),
            root / "bin" / "python",
        )

    def test_supported_python_range(self):
        for minor in (10, 11, 12, 13):
            self.assertTrue(
                quickstart.is_supported_version(SimpleNamespace(major=3, minor=minor))
            )
        for version in ((3, 9), (3, 14), (4, 0)):
            self.assertFalse(
                quickstart.is_supported_version(
                    SimpleNamespace(major=version[0], minor=version[1])
                )
            )

    def test_missing_venv_creates_installs_and_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_checkout(root)
            runner = FakeRunner(root)

            result = quickstart.run_quickstart(
                project_root=root,
                base_python=Path("python"),
                version_info=VERSION_310,
                os_name="nt",
                runner=runner,
            )

            self.assertEqual(result, 0)
            self.assertEqual(len(runner.commands), 4)
            self.assertEqual(runner.commands[0][1:3], ["-m", "venv"])
            self.assertIn("pip", runner.commands[2])
            self.assertEqual(runner.commands[3][-1], "+experiment=smoke")
            stamp = root / ".venv" / quickstart.STAMP_NAME
            self.assertEqual(
                stamp.read_text(encoding="ascii").strip(),
                quickstart.requirements_digest(root / "requirements-smoke.txt"),
            )

    def test_matching_hash_skips_pip_and_does_not_print_api_key(self):
        secret = "sk-" + "x" * 30
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_checkout(root)
            python_path = quickstart.venv_python(root / ".venv", "nt")
            python_path.parent.mkdir(parents=True)
            python_path.touch()
            digest = quickstart.requirements_digest(root / "requirements-smoke.txt")
            (root / ".venv" / quickstart.STAMP_NAME).write_text(
                digest + "\n", encoding="ascii"
            )
            runner = FakeRunner(root)
            output = io.StringIO()

            with (
                mock.patch.dict(os.environ, {"LLM_API_KEY": secret}),
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(output),
            ):
                result = quickstart.run_quickstart(
                    project_root=root,
                    version_info=VERSION_310,
                    os_name="nt",
                    runner=runner,
                )

            self.assertEqual(result, 0)
            self.assertFalse(any("pip" in command for command in runner.commands))
            self.assertNotIn(secret, output.getvalue())

    def test_changed_hash_reinstalls_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_checkout(root)
            python_path = quickstart.venv_python(root / ".venv", "nt")
            python_path.parent.mkdir(parents=True)
            python_path.touch()
            (root / ".venv" / quickstart.STAMP_NAME).write_text(
                "outdated\n", encoding="ascii"
            )
            runner = FakeRunner(root)

            result = quickstart.run_quickstart(
                project_root=root,
                version_info=VERSION_310,
                os_name="nt",
                runner=runner,
            )

            self.assertEqual(result, 0)
            self.assertTrue(any("pip" in command for command in runner.commands))

    def test_failures_propagate_without_running_later_stages(self):
        for stage, expected in (("venv", 7), ("pip", 8), ("smoke", 9)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                make_checkout(root)
                runner = FakeRunner(root, failures={stage: expected})
                if stage != "venv":
                    python_path = quickstart.venv_python(root / ".venv", "nt")
                    python_path.parent.mkdir(parents=True)
                    python_path.touch()

                result = quickstart.run_quickstart(
                    project_root=root,
                    base_python=Path("python"),
                    version_info=VERSION_310,
                    os_name="nt",
                    runner=runner,
                )

                self.assertEqual(result, expected)
                if stage in {"venv", "pip"}:
                    self.assertFalse(
                        any(command[-1] == "+experiment=smoke" for command in runner.commands)
                    )

    def test_incompatible_or_broken_environment_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_checkout(root)
            (root / ".venv").mkdir()
            runner = FakeRunner(root)

            result = quickstart.run_quickstart(
                project_root=root,
                version_info=VERSION_310,
                os_name="nt",
                runner=runner,
            )

            self.assertEqual(result, 2)
            self.assertEqual(runner.commands, [])

        runner = mock.Mock()
        result = quickstart.run_quickstart(
            project_root=Path.cwd(),
            version_info=SimpleNamespace(major=3, minor=14),
            runner=runner,
        )
        self.assertEqual(result, 2)
        runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
