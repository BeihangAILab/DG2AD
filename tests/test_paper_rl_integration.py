import tempfile
import unittest
import importlib
from pathlib import Path

import torch

from tests.synthetic_domains import synthetic_instance

from src.core.structure import END
from src.core.contracts import (
    CandidateResult,
    PipelineSample,
    PolicyUpdateStats,
    RewardSpec,
)
from src.core.execution import RewardBatchEvaluator
from src.core.operators import RunOperatorPool
from src.core.contracts import standardize_candidate_rewards
from src.core.trainer import PaperRLTrainer
from src.core.reporting import TrainingReporter


ROOT = Path(__file__).resolve().parents[1]


class FakeLLM:
    chat_calls = 0

    def __init__(self):
        self.calls_used = 0
        self.max_calls = 100

    def chat(self, messages, temperature=None):
        type(self).chat_calls += 1
        self.calls_used += 1
        raise RuntimeError("offline fake")

    def state_dict(self):
        return {"calls_used": self.calls_used, "max_calls": self.max_calls}

    def load_state_dict(self, state):
        self.calls_used = state["calls_used"]


class FakeDomain:
    __name__ = "fake_domain"

    @staticmethod
    def load_instance_group(group):
        return [
            {"name": f"instance-{index}", "upper_bound": 10.0, "size": index} for index in range(6)
        ]

    @staticmethod
    def get_reward_spec(env_data):
        return RewardSpec("min", 10.0, 10.0)

    @staticmethod
    def describe_instance(env_data):
        return f"name={env_data['name']};size={env_data['size']}"

    @staticmethod
    def get_initial_pipeline():
        return ["initialization|init"]


class FakePolicy:
    update_calls = 0

    def __init__(self, config, graph, project_root):
        self.torch = torch
        self.alias_by_action = {"initialization|init": "<A000>", END: "<END>"}
        self.graph = graph
        self.model_name = "fake"
        self.revision = "main"

    def set_graph(self, graph):
        self.graph = graph

    def sample(self, context, graph, max_pipeline_length, temperature, greedy=False):
        return PipelineSample(["initialization|init"])

    def beam_search(self, context, graph, max_pipeline_length, temperature, beam_size):
        return [PipelineSample(["initialization|init"])]

    def update(self, results):
        type(self).update_calls += 1
        return PolicyUpdateStats(loss=0.0, grad_norm=0.0)

    def training_state(self):
        return {"adapter": {}, "optimizer": {}, "model_name": "fake", "revision": "main"}

    def load_training_state(self, state, load_optimizer=True):
        return None

    def save_adapter(self, directory):
        adapter = Path(directory) / "adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "fake.safetensors").write_bytes(b"fake")


class FakeEvaluator:
    calls = []
    instance_batches = []

    def __init__(self, domain, slots, steps, timeout, failure, epsilon):
        self.epsilon = epsilon

    def evaluate(self, samples, instances, standardize=True, evaluation_seed=0):
        type(self).calls.append((len(samples), len(instances), standardize))
        type(self).instance_batches.append(tuple(instance["name"] for instance in instances))
        results = []
        for index, sample in enumerate(samples):
            result = CandidateResult(sample=sample, scores=[10.0] * len(instances))
            result.raw_reward = float(index)
            results.append(result)
        if standardize:
            standardize_candidate_rewards(results, self.epsilon)
        return results


class RecordingReporter(TrainingReporter):
    def __init__(self):
        self.events = []

    def __getattribute__(self, name):
        if name in {
            "initialization",
            "model_ready",
            "generation_start",
            "sampling_complete",
            "candidates",
            "policy_update",
            "validation",
            "evolution",
            "checkpoint",
            "complete",
        }:

            def record(*args, **kwargs):
                self.events.append(name)

            return record
        return super().__getattribute__(name)


class FullTrainerIntegrationTests(unittest.TestCase):
    @staticmethod
    def config(cache_dir, generations=2, resume=False, checkpoints=False):
        return {
            "problem": {"target_group": "fake"},
            "engine": {
                "seed": 0,
                "run_tier": "quick",
                "generations": generations,
                "candidates_per_generation": 2,
                "shared_instances_per_generation": 2,
                "pipeline_steps": 1,
                "pipeline_timeout": 1,
                "failure_reward": -5,
                "reward_epsilon": 1e-8,
            },
            "dataset": {
                "train_ratio": 0.6,
                "validation_ratio": 0.2,
                "test_ratio": 0.2,
                "split_seed": 0,
            },
            "policy": {"cache_dir": str(cache_dir)},
            "credit": {"order": "first", "alpha": 0.1, "aggregation_epsilon": 1e-8},
            "initialization": {
                "retries": 0,
                "max_operator_count": {"fallback": 1, "min": 1, "max": 1, "llm_select": False},
                "max_pipeline_length": {"fallback": 1, "min": 1, "max": 1, "llm_select": False},
                "sampling_temperature": {
                    "fallback": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "llm_select": False,
                },
            },
            "evolution": {
                "operator_interval": 1,
                "operator_changes_per_event": 1,
                "graph_interval": 1,
                "max_added_edges": 2,
                "max_deleted_edges": 2,
                "max_versions_per_node": 10,
                "min_dominance_trials": 5,
                "dominance_margin": 0.5,
            },
            "validation": {"interval": 5, "beam_size": 4},
            "checkpoint": {"resume": resume, "save_last": checkpoints, "save_best": checkpoints},
        }

    def test_two_generations_use_shared_n_by_k_batch_and_one_update_each(self):
        FakePolicy.update_calls = 0
        FakeEvaluator.calls = []
        FakeEvaluator.instance_batches = []
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            temp = Path(temporary)
            source = temp / "source" / "initialization" / "init"
            source.mkdir(parents=True)
            (source / "v1.py").write_text(
                "def run(env_data, state, calc_makespan_fn):\n    return state\n",
                encoding="utf-8",
            )
            config = self.config(temp / "model_cache")
            reporter = RecordingReporter()
            trainer = PaperRLTrainer(
                FakeDomain,
                "fake optimisation problem",
                config,
                str(temp / "source"),
                str(temp / "run"),
                str(ROOT),
                policy_factory=FakePolicy,
                evaluator_factory=FakeEvaluator,
                llm_client=FakeLLM(),
                reporter=reporter,
            )
            result = trainer.run()
            self.assertTrue((temp / "run" / "manifest.json").is_file())
            self.assertTrue((temp / "run" / "events.jsonl").is_file())
            self.assertTrue((temp / "run" / "result.json").is_file())
            manifest = __import__("json").loads(
                (temp / "run" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["schema_version"], 3)
            self.assertEqual(len(manifest["evolution_batch"]), 2)

        training_calls = [call for call in FakeEvaluator.calls if call == (2, 2, True)]
        self.assertEqual(len(training_calls), 2)
        training_batches = [
            batch
            for call, batch in zip(FakeEvaluator.calls, FakeEvaluator.instance_batches)
            if call == (2, 2, True)
        ]
        self.assertEqual(training_batches[0], training_batches[1])
        self.assertEqual(FakePolicy.update_calls, 2)
        self.assertEqual(result["markov_total_trials"], 4)
        self.assertEqual(len(result["events"]), 2)
        self.assertEqual(result["generations_planned"], 2)
        self.assertEqual(result["generations_completed"], 2)
        self.assertNotIn("wall_time_limited", result)
        first = result["events"][0]
        self.assertIn("operator_evolution", first)
        self.assertIn("graph_evolution", first)
        self.assertIn("gradient_norm", first)
        self.assertEqual(
            reporter.events[:6],
            [
                "initialization",
                "model_ready",
                "generation_start",
                "sampling_complete",
                "candidates",
                "policy_update",
            ],
        )
        self.assertEqual(reporter.events[-1], "complete")

    def test_training_runs_exactly_the_configured_number_of_generations(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            temp = Path(temporary)
            source = temp / "source" / "initialization" / "init"
            source.mkdir(parents=True)
            (source / "v1.py").write_text(
                "def run(env_data, state, calc_makespan_fn):\n    return state\n",
                encoding="utf-8",
            )
            config = self.config(temp / "model_cache", generations=3)
            config["validation"]["interval"] = 1
            result = PaperRLTrainer(
                FakeDomain,
                "fake optimisation problem",
                config,
                str(temp / "source"),
                str(temp / "run"),
                str(ROOT),
                policy_factory=FakePolicy,
                evaluator_factory=FakeEvaluator,
                llm_client=FakeLLM(),
                reporter=RecordingReporter(),
            ).run()
            self.assertEqual(result["generations_planned"], 3)
            self.assertEqual(result["generations_completed"], 3)
            self.assertEqual(len(result["events"]), 3)
            self.assertNotIn("wall_time_limited", result)
            self.assertIsNotNone(result["test_gap_percent"])

    def test_resume_reads_last_before_any_initialization_llm_call(self):
        FakeLLM.chat_calls = 0
        FakeEvaluator.calls = []
        FakeEvaluator.instance_batches = []
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            temp = Path(temporary)
            source = temp / "source" / "initialization" / "init"
            source.mkdir(parents=True)
            (source / "v1.py").write_text(
                "def run(env_data, state, calc_makespan_fn):\n    return state\n",
                encoding="utf-8",
            )
            run_dir = temp / "run"
            first = PaperRLTrainer(
                FakeDomain,
                "fake",
                self.config(temp / "cache", 1, False, True),
                str(temp / "source"),
                str(run_dir),
                str(ROOT),
                policy_factory=FakePolicy,
                evaluator_factory=FakeEvaluator,
                llm_client=FakeLLM(),
            )
            first.run()
            first_manifest = __import__("json").loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            FakeLLM.chat_calls = 0
            resumed = PaperRLTrainer(
                FakeDomain,
                "fake",
                self.config(temp / "cache", 2, True, True),
                str(temp / "source"),
                str(run_dir),
                str(ROOT),
                policy_factory=FakePolicy,
                evaluator_factory=FakeEvaluator,
                llm_client=FakeLLM(),
            )
            result = resumed.run()
            resumed_manifest = __import__("json").loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                resumed_manifest["evolution_batch"], first_manifest["evolution_batch"]
            )
            training_batches = [
                batch
                for call, batch in zip(FakeEvaluator.calls, FakeEvaluator.instance_batches)
                if call == (2, 2, True)
            ]
            self.assertEqual(len(training_batches), 2)
            self.assertEqual(training_batches[0], training_batches[1])
        self.assertEqual(FakeLLM.chat_calls, 0)
        self.assertEqual(result["generations_completed"], 2)
        self.assertEqual(result["events"][0]["generation"], 2)

    def test_resume_rejects_v2_without_a_fixed_batch_record(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            temp = Path(temporary)
            source = temp / "source" / "initialization" / "init"
            source.mkdir(parents=True)
            (source / "v1.py").write_text(
                "def run(env_data, state, calc_makespan_fn):\n    return state\n",
                encoding="utf-8",
            )
            run_dir = temp / "run"
            PaperRLTrainer(
                FakeDomain,
                "fake",
                self.config(temp / "cache", 1, False, True),
                str(temp / "source"),
                str(run_dir),
                str(ROOT),
                policy_factory=FakePolicy,
                evaluator_factory=FakeEvaluator,
                llm_client=FakeLLM(),
            ).run()
            checkpoint_manifest = run_dir / "checkpoints" / "last" / "manifest.json"
            manifest = __import__("json").loads(checkpoint_manifest.read_text(encoding="utf-8"))
            manifest["schema_version"] = 2
            manifest.pop("evolution_batch", None)
            checkpoint_manifest.write_text(__import__("json").dumps(manifest), encoding="utf-8")
            trainer = PaperRLTrainer(
                FakeDomain,
                "fake",
                self.config(temp / "cache", 2, True, True),
                str(temp / "source"),
                str(run_dir),
                str(ROOT),
                policy_factory=FakePolicy,
                evaluator_factory=FakeEvaluator,
                llm_client=FakeLLM(),
            )
            with self.assertRaisesRegex(RuntimeError, "fixed run-scoped evolution batches"):
                trainer.run()

    def test_real_router_worker_executes_an_isolated_run_pool(self):
        domain = importlib.import_module("src.problems.tsp.domain_evaluator")
        nodes = domain.get_initial_pipeline()
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            pool = RunOperatorPool.create_from_source(
                ROOT / "src" / "problems" / "tsp" / "slots",
                Path(temporary) / "operator_pool",
                nodes,
            )
            sample = PipelineSample(
                nodes=nodes,
                implementations=[(*node.split("|", 1), "v1") for node in nodes],
            )
            instances = [synthetic_instance("tsp")]
            outcome = RewardBatchEvaluator(domain, str(pool.root), 1, 20.0, -5.0, 1e-8).evaluate(
                [sample], [instances[0]], standardize=False
            )[0]
            failed = RewardBatchEvaluator(domain, str(pool.root), 1, 20.0, -5.0, 1e-8).evaluate(
                [sample],
                [instances[0], {"name": "invalid"}],
                standardize=False,
            )[0]
        self.assertFalse(outcome.failed, outcome.failure_reason)
        self.assertEqual(len(outcome.scores), 1)
        self.assertTrue(failed.failed)
        self.assertEqual(failed.raw_reward, -5.0)


if __name__ == "__main__":
    unittest.main()
