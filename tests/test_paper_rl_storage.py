import os
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src.core.checkpoint import CheckpointManager
from src.core.credit import TransitionCreditStore
from src.core.structure import DirectedOperatorGraph, END, START
from src.core.operators import RunOperatorPool
from src.core.configuration import (
    configure_huggingface_cache,
    resolve_project_path,
)


ROOT = Path(__file__).resolve().parents[1]
VALID_CODE = "def run(env_data, state, calc_makespan_fn):\n    return state\n"


class OperatorPoolTests(unittest.TestCase):
    def test_add_delete_archive_and_replace_rollback_leave_source_unchanged(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            source = root / "source"
            node = source / "initialization" / "init"
            node.mkdir(parents=True)
            (node / "v1.py").write_text(VALID_CODE, encoding="utf-8")
            source_bytes = (node / "v1.py").read_bytes()
            pool = RunOperatorPool.create_from_source(
                source, root / "run" / "operator_pool", ["initialization|init"]
            )
            v2 = pool.add_code("initialization|init", VALID_CODE + "\n# v2\n")
            archived = pool.archive("initialization|init", "v1", "test")
            self.assertTrue(archived.is_file())
            self.assertEqual(pool.available_versions("initialization|init"), [v2])

            with mock.patch.object(pool, "archive", side_effect=RuntimeError("stop")):
                with self.assertRaises(RuntimeError):
                    pool.replace("initialization|init", v2, VALID_CODE + "\n# v3\n")
            self.assertEqual(pool.available_versions("initialization|init"), [v2])
            self.assertEqual((node / "v1.py").read_bytes(), source_bytes)

    def test_restore_snapshot_retries_a_transient_windows_directory_lock(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            source = root / "source" / "initialization" / "init"
            source.mkdir(parents=True)
            (source / "v1.py").write_text(VALID_CODE, encoding="utf-8")
            pool = RunOperatorPool.create_from_source(
                root / "source", root / "run" / "operator_pool", ["initialization|init"]
            )
            snapshot = root / "snapshot"
            shutil.copytree(pool.root, snapshot)
            pool.add_code("initialization|init", VALID_CODE + "\n# transient\n")

            real_replace = os.replace
            calls = 0

            def transient_replace(source_path, destination_path):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise PermissionError("temporary scanner lock")
                return real_replace(source_path, destination_path)

            with mock.patch(
                "src.core.operators.os.replace",
                side_effect=transient_replace,
            ):
                pool.restore_snapshot(snapshot)
            self.assertGreaterEqual(calls, 2)
            self.assertEqual(pool.available_versions("initialization|init"), ["v1"])


class FakeCheckpointPolicy:
    def __init__(self):
        self.torch = torch
        self.alias_by_action = {"initialization|init": "<A000>", END: "<END>"}
        self.value = torch.tensor([1.0])
        self.graph = None

    def training_state(self):
        return {"adapter": {"value": self.value.clone()}, "optimizer": {"step": 3}}

    def load_training_state(self, state, load_optimizer=True):
        self.value = state["adapter"]["value"].clone()

    def save_adapter(self, directory):
        adapter = Path(directory) / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"fake")

    def set_graph(self, graph):
        self.graph = graph


class CheckpointTests(unittest.TestCase):
    def test_last_round_trip_restores_graph_pool_credit_policy_and_rng(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            source = root / "source" / "initialization" / "init"
            source.mkdir(parents=True)
            (source / "v1.py").write_text(VALID_CODE, encoding="utf-8")
            pool = RunOperatorPool.create_from_source(
                root / "source", root / "run" / "operator_pool", ["initialization|init"]
            )
            graph = DirectedOperatorGraph(
                {"initialization|init"},
                {(START, "initialization|init"), ("initialization|init", END)},
                {"initialization|init"},
                {"initialization|init"},
            )
            credit = TransitionCreditStore()
            key = (credit.START_IMPLEMENTATION, "initialization|init|v1")
            credit.update([key], [(None, "initialization|init")], 2.0)
            policy = FakeCheckpointPolicy()
            rng = random.Random(123)
            manager = CheckpointManager(root / "run")
            cache_dir = pool.root / "initialization" / "init" / "__pycache__"
            cache_dir.mkdir()
            (cache_dir / "v1.pyc").write_bytes(b"cache")
            manager.save(
                "last",
                4,
                graph,
                credit,
                policy,
                pool,
                {"dataset": "fake"},
                {"best_validation": 1.2},
                {"calls_used": 7, "max_calls": 10},
                rng,
            )
            expected_next = rng.random()
            policy.value.fill_(9)
            pool.add_code("initialization|init", VALID_CODE + "\n# extra\n")
            restored = manager.load("last", policy, pool, rng, load_optimizer=True)
            self.assertEqual(restored["checkpoint"]["generation"], 4)
            self.assertAlmostEqual(restored["credit"].get(key).q, 0.2)
            self.assertEqual(policy.value.item(), 1.0)
            self.assertEqual(pool.available_versions("initialization|init"), ["v1"])
            self.assertAlmostEqual(rng.random(), expected_next)
            self.assertTrue((manager.path("last") / "aliases.json").is_file())
            checkpoint = json.loads(
                (manager.path("last") / "checkpoint.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["schema_version"], 3)
            self.assertFalse(any((manager.path("last") / "operator_pool").rglob("*.pyc")))

            checkpoint["schema_version"] = 1
            (manager.path("last") / "checkpoint.json").write_text(
                json.dumps(checkpoint), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "v3"):
                manager.load("last", policy, pool, rng, load_optimizer=True)


class PortableStorageTests(unittest.TestCase):
    def test_relative_paths_resolve_from_project_root(self):
        resolved = resolve_project_path(ROOT, ".model_cache/huggingface")
        self.assertEqual(resolved, (ROOT / ".model_cache/huggingface").resolve())

    def test_project_local_cache_sets_every_download_environment_variable(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            cache = configure_huggingface_cache(ROOT, Path(temporary) / "hf")
            for name in (
                "HF_HOME",
                "HF_HUB_CACHE",
                "HF_XET_CACHE",
                "TRANSFORMERS_CACHE",
                "TEMP",
                "TMP",
                "TMPDIR",
            ):
                self.assertTrue(Path(os.environ[name]).is_relative_to(cache))


if __name__ == "__main__":
    unittest.main()
