import builtins
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hydra import compose, initialize_config_dir

import main as entrypoint
from src.core.reporting import ConsoleReporter
from src.utils import llm_client


ROOT = Path(__file__).resolve().parents[1]


class OfflineSmokeTests(unittest.TestCase):
    def test_preset_selects_offline_smoke_action(self):
        with initialize_config_dir(config_dir=str((ROOT / "cfg").resolve()), version_base=None):
            cfg = compose(
                config_name="config",
                overrides=["+experiment=smoke"],
            )
        self.assertEqual(cfg.problem.name, "fssp")
        self.assertEqual(cfg.engine.action, "smoke")
        self.assertEqual(cfg.engine.pipeline_steps, 1)
        self.assertEqual(cfg.engine.evaluator_workers, 1)

    def test_real_smoke_does_not_load_models_or_call_network(self):
        with initialize_config_dir(config_dir=str((ROOT / "cfg").resolve()), version_base=None):
            cfg = compose(
                config_name="config",
                overrides=["+experiment=smoke"],
            )

        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in {"torch", "transformers", "peft"}:
                raise AssertionError(f"smoke imported training dependency {name}")
            return original_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            reporter = ConsoleReporter(run_dir, color="never")
            create_client = mock.Mock(side_effect=AssertionError("smoke created an LLM client"))
            try:
                with (
                    mock.patch.object(llm_client, "create_client", create_client),
                    mock.patch("builtins.__import__", side_effect=guarded_import),
                    mock.patch(
                        "urllib.request.urlopen",
                        side_effect=AssertionError("smoke attempted network access"),
                    ),
                    mock.patch(
                        "socket.create_connection",
                        side_effect=AssertionError("smoke attempted network access"),
                    ),
                ):
                    result = entrypoint.dispatch_run(cfg, run_dir, reporter)
            finally:
                reporter.close()

            self.assertEqual(result["status"], "pass")
            create_client.assert_not_called()
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["offline"])
            self.assertEqual(
                {item[-1] for item in manifest["pipeline"]["implementations"]},
                {"v1"},
            )
            self.assertTrue((run_dir / "result.json").is_file())
            self.assertFalse((run_dir / "checkpoints").exists())
            self.assertFalse(any((run_dir / "operator_pool").rglob("*.pyc")))
            self.assertIn(
                "[SMOKE] PASS",
                (run_dir / "run.log").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
