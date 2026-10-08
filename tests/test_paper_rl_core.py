import math
import importlib
import copy
import random
import tempfile
import unittest
from types import MethodType, SimpleNamespace
from unittest import mock
from pathlib import Path

import torch
from src.core.credit import TransitionCreditStore
from src.core.dataset import sample_instances, split_domain_instances
from src.core.execution import RewardBatchEvaluator
from src.core.evolution import GraphEvolver, RuleBasedOperatorEvolver
from src.core.operators import RunOperatorPool
from src.core.structure import DirectedOperatorGraph, END, START
from src.core.contracts import (
    CandidateResult,
    GraphDelta,
    PipelineSample,
    PolicyUpdateStats,
    RewardSpec,
)
from src.core.policy import PipelinePolicy
from src.core.structure import PaperInitializer
from tests.synthetic_domains import synthetic_instance
from src.core.contracts import (
    candidate_reward,
    normalized_gap,
    standardize_candidate_rewards,
)


class DirectedGraphTests(unittest.TestCase):
    def graph(self):
        return DirectedOperatorGraph(
            {"initialization|i", "operator|a", "operator|b"},
            {
                (START, "initialization|i"),
                ("initialization|i", "operator|a"),
                ("operator|a", "operator|b"),
                ("operator|b", END),
            },
            {"initialization|i"},
            {"operator|b"},
        )

    def test_illegal_sentinel_and_missing_exit_are_rejected(self):
        with self.assertRaises(ValueError):
            DirectedOperatorGraph(
                {"initialization|i", "operator|a"},
                {(START, "operator|a"), ("operator|a", END)},
                {"initialization|i"},
                {"operator|a"},
            )
        with self.assertRaises(ValueError):
            DirectedOperatorGraph(
                {"initialization|i"},
                {(START, "initialization|i")},
                {"initialization|i"},
                {"initialization|i"},
            )
        with self.assertRaises(ValueError):
            DirectedOperatorGraph(
                {"initialization|i", "initialization|j", "operator|a"},
                {
                    (START, "initialization|i"),
                    ("initialization|i", "initialization|j"),
                    ("initialization|j", "operator|a"),
                    ("operator|a", END),
                },
                {"initialization|i"},
                {"operator|a"},
            )

    def test_delete_last_path_rolls_back(self):
        graph = self.graph()
        before = graph.to_dict()
        with self.assertRaises(ValueError):
            graph.apply_delta(GraphDelta(delete_edges=[("operator|b", END)]), 2, 2)
        self.assertEqual(graph.to_dict(), before)

    def test_cycles_repeated_nodes_and_length_mask(self):
        graph = self.graph().apply_delta(GraphDelta(add_edges=[("operator|b", "operator|a")]), 2, 2)
        self.assertIn("operator|a", graph.valid_next_actions("operator|b", 2))
        self.assertNotIn("operator|a", graph.valid_next_actions("operator|b", 1))
        self.assertIn(END, graph.valid_next_actions("operator|b", 0))
        self.assertEqual(
            graph.route_through_edge(("operator|b", "operator|a")),
            ["initialization|i", "operator|a", "operator|b", "operator|a", "operator|b"],
        )

    def test_runtime_validation_failure_is_transactional(self):
        graph = self.graph()
        before = graph.to_dict()
        with self.assertRaises(ValueError):
            graph.apply_delta(
                GraphDelta(add_edges=[("operator|b", "operator|a")]),
                2,
                2,
                runtime_validation_validator=lambda edge, candidate: False,
            )
        self.assertEqual(graph.to_dict(), before)


class RewardTests(unittest.TestCase):
    def test_signed_min_max_and_negative_objectives(self):
        self.assertAlmostEqual(normalized_gap(12.0, RewardSpec("min", 10.0, 10.0), 0), 0.2)
        self.assertAlmostEqual(normalized_gap(12.0, RewardSpec("max", 10.0, 10.0), 0), -0.2)
        self.assertAlmostEqual(normalized_gap(-12.0, RewardSpec("min", -10.0, 10.0), 0), -0.2)
        self.assertAlmostEqual(
            candidate_reward(
                [12.0, 8.0],
                [RewardSpec("min", 10.0, 10.0)] * 2,
                0,
            ),
            0.0,
        )

    def test_3dclp_fill_space_keeps_negative_gap(self):
        domain_evaluator = importlib.import_module("src.problems.3dclp.domain_evaluator")

        spec = domain_evaluator.get_reward_spec({"fill_upper_bound": 80.0})
        self.assertAlmostEqual(normalized_gap(15.0, spec, 0), -0.0625)

    def test_population_standardization_and_zero_variance(self):
        candidates = [CandidateResult(PipelineSample(["a"]), raw_reward=x) for x in (1, 2, 3)]
        mean, std = standardize_candidate_rewards(candidates, 0)
        self.assertEqual(mean, 2)
        self.assertAlmostEqual(std, math.sqrt(2 / 3))
        equal = [CandidateResult(PipelineSample(["a"]), raw_reward=2) for _ in range(3)]
        _, std = standardize_candidate_rewards(equal)
        self.assertEqual(std, 0)
        self.assertEqual([item.normalized_reward for item in equal], [0, 0, 0])


class RewardBatchAlignmentTests(unittest.TestCase):
    def test_out_of_order_futures_keep_scores_aligned_with_instances(self):
        class Future:
            def __init__(self, env_data):
                self.env_data = env_data

            def result(self):
                return (
                    [self.env_data["name"]],
                    [],
                    self.env_data["score"],
                    None,
                    None,
                    0.01,
                    {},
                    None,
                )

            def cancel(self):
                return False

        class Executor:
            _processes = {}

            def submit(self, _function, *args):
                return Future(args[1])

            def shutdown(self, **_kwargs):
                return None

        domain = SimpleNamespace(
            __name__="fake_reward_domain",
            get_reward_spec=lambda env: RewardSpec("min", env["reference"], env["reference"]),
        )
        evaluator = RewardBatchEvaluator(domain, "unused", 1, 5.0, -5.0, 0.0, max_workers=2)
        instances = [
            {"name": "first", "score": 11.0, "reference": 10.0},
            {"name": "second", "score": 22.0, "reference": 20.0},
        ]
        with (
            mock.patch(
                "src.core.execution.concurrent.futures.ProcessPoolExecutor",
                return_value=Executor(),
            ) as executor_factory,
            mock.patch(
                "src.core.execution.concurrent.futures.as_completed",
                side_effect=lambda futures, timeout: list(reversed(list(futures))),
            ),
        ):
            result = evaluator.evaluate(
                [PipelineSample(["initialization|i"])],
                instances,
                standardize=False,
            )[0]

        self.assertEqual(
            executor_factory.call_args.kwargs["mp_context"].get_start_method(),
            "spawn",
        )
        self.assertFalse(result.failed)
        self.assertEqual(result.scores, [11.0, 22.0])
        self.assertEqual(result.representative_sequence, ["first"])
        self.assertAlmostEqual(result.raw_reward, -0.1)


class MarkovCreditTests(unittest.TestCase):
    def test_context_orders_match_the_paper_ablation(self):
        history = ["a|v1", "b|v1"]
        self.assertEqual(TransitionCreditStore(order="zero").context_key(history, "c|v1"), ("c|v1",))
        self.assertEqual(
            TransitionCreditStore(order="first").context_key(history, "c|v1"),
            ("b|v1", "c|v1"),
        )
        self.assertEqual(
            TransitionCreditStore(order="second").context_key(history, "c|v1"),
            ("a|v1", "b|v1", "c|v1"),
        )
        self.assertEqual(
            TransitionCreditStore(order="full").context_key(history, "c|v1"),
            (TransitionCreditStore.START_IMPLEMENTATION, "a|v1", "b|v1", "c|v1"),
        )
        self.assertEqual(
            TransitionCreditStore(order="second").context_key([], "c|v1"),
            (
                TransitionCreditStore.START_IMPLEMENTATION,
                TransitionCreditStore.START_IMPLEMENTATION,
                "c|v1",
            ),
        )

    def test_ema_and_edge_aggregation_round_trip_for_zero_order(self):
        store = TransitionCreditStore(alpha=0.1, order="zero")
        key = ("node|v1",)
        store.update([key], [("previous", "node")], 2.0)
        self.assertAlmostEqual(store.get(key).q, 0.2)
        self.assertAlmostEqual(store.aggregate_edges()[("previous", "node")].q, 0.2)
        restored = TransitionCreditStore.from_dict(store.to_dict())
        self.assertEqual(restored.order, "zero")
        self.assertEqual(restored.edge_context_trials, store.edge_context_trials)

    def test_operator_credit_is_weighted_by_implementation_activations(self):
        store = TransitionCreditStore(alpha=1.0, order="zero")
        first = ("slot|operator|v1",)
        second = ("slot|operator|v2",)
        store.update([first], [(None, "slot|operator")], 1.0)
        for _ in range(3):
            store.update([second], [(None, "slot|operator")], -1.0)
        operator = store.aggregate_operators({"slot|operator": ["v1", "v2"]})[
            "slot|operator"
        ]
        self.assertEqual(operator.trials, 4)
        self.assertAlmostEqual(operator.q, -0.5)

    def test_boltzmann_uses_q_over_temperature(self):
        store = TransitionCreditStore()
        key1 = ("prev", "node|v1")
        key2 = ("prev", "node|v2")
        store.get(key1).q = 1.0
        store.get(key2).q = 0.0

        class CapturingRandom:
            weights = None

            def choices(self, population, weights, k):
                self.weights = weights
                return [0]

        rng = CapturingRandom()
        version, key = store.choose_version(["prev"], "node", ["v1", "v2"], 0.5, rng)
        self.assertEqual((version, key), ("v1", key1))
        self.assertAlmostEqual(rng.weights[0] / rng.weights[1], math.exp(2.0))


class CreditGuidedEvolutionTests(unittest.TestCase):
    def test_graph_evolution_only_replaces_a_negative_lowest_edge(self):
        nodes = {"initialization|i", "operator|a", "operator|b", "operator|c"}
        graph = DirectedOperatorGraph(
            nodes,
            {
                (START, "initialization|i"),
                ("initialization|i", "operator|a"),
                ("operator|a", "operator|b"),
                ("operator|a", "operator|c"),
                ("operator|b", END),
                ("operator|c", END),
            },
            {"initialization|i"},
            {"operator|b", "operator|c"},
        )

        class LLM:
            calls = 0

            def chat(self, messages):
                self.calls += 1
                return (
                    '{"delete_edges":[["operator|a","operator|b"]],'
                    '"add_edges":[["operator|b","operator|c"]]}',
                    {},
                )

        credit = TransitionCreditStore(alpha=1.0)
        credit.update(
            [("operator|a|v1", "operator|b|v1")],
            [("operator|a", "operator|b")],
            -1.0,
        )
        llm = LLM()
        evolved, delta = GraphEvolver({"structure": {"name": "dg"}}, llm, "domain").evolve(
            graph, credit, lambda edge, candidate: True
        )
        self.assertEqual(llm.calls, 1)
        self.assertIsNotNone(delta)
        self.assertNotIn(("operator|a", "operator|b"), evolved.edges)
        self.assertIn(("operator|b", "operator|c"), evolved.edges)

        positive = TransitionCreditStore(alpha=1.0)
        positive.update(
            [("operator|a|v1", "operator|b|v1")],
            [("operator|a", "operator|b")],
            1.0,
        )
        unchanged, delta = GraphEvolver(
            {"structure": {"name": "dg"}}, llm, "domain"
        ).evolve(graph, positive, lambda edge, candidate: True)
        self.assertIsNone(delta)
        self.assertEqual(unchanged.to_dict(), graph.to_dict())
        self.assertEqual(llm.calls, 1)

    def test_operator_evolution_uses_operator_credit_sign_and_rolls_back(self):
        code = "def run(env_data, state, calc_makespan_fn):\n    return state\n"
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            root = Path(temporary)
            source = root / "source" / "operator" / "a"
            source.mkdir(parents=True)
            (source / "v1.py").write_text(code, encoding="utf-8")
            pool = RunOperatorPool.create_from_source(
                root / "source", root / "pool", ["operator|a"]
            )
            pool.add_code("operator|a", code + "\n# second\n")
            evolver = RuleBasedOperatorEvolver(
                {"evolution": {"max_versions_per_node": 2}, "engine": {}},
                object(),
                "domain",
                SimpleNamespace(__name__="domain"),
                {},
                str(root),
            )
            result = evolver.evolve(pool, TransitionCreditStore())
            self.assertEqual(result.operation, "DELETE")
            self.assertEqual(len(pool.available_versions("operator|a")), 1)

            credit = TransitionCreditStore(alpha=1.0)
            active_version = pool.available_versions("operator|a")[0]
            implementation = credit.implementation_id("operator|a", active_version)
            credit.update(
                [(credit.START_IMPLEMENTATION, implementation)],
                [(None, "operator|a")],
                -1.0,
            )
            evolver._generate = lambda *args: code + "\n# generated\n"
            result = evolver.evolve(pool, credit, combination_validator=lambda *args: False)
            self.assertIsNone(result)
            self.assertEqual(pool.available_versions("operator|a"), [active_version])


class DatasetTests(unittest.TestCase):
    def test_missing_benchmark_reports_the_preparation_command(self):
        domain = SimpleNamespace(
            __name__="src.problems.fssp.domain_evaluator",
            load_instance_group=lambda _group: [],
        )
        with self.assertRaisesRegex(ValueError, r"prepare_data\.py --problem fssp"):
            split_domain_instances(domain, "tai20_5")

    def test_split_is_deterministic_disjoint_and_sampling_has_no_replacement(self):
        instances = [{"name": f"i{index}", "upper_bound": 1.0} for index in range(10)]
        domain = SimpleNamespace(
            load_instance_group=lambda group: list(reversed(instances)),
            describe_instance=lambda env: env["name"],
        )
        first = split_domain_instances(domain, "g", seed=7)
        second = split_domain_instances(domain, "g", seed=7)

        def ids(values):
            return [item["name"] for item in values]

        self.assertEqual(ids(first.train), ids(second.train))
        groups = [set(ids(values)) for values in (first.train, first.validation, first.test)]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        chosen = sample_instances(first.train, 3, random.Random(0))
        self.assertEqual(len({item["name"] for item in chosen}), 3)
        self.assertTrue(all("content_sha256" in item for item in first.manifest["train"]))

    def test_all_twelve_domain_splits_have_unique_disjoint_manifest_ids(self):
        problems = (
            "3dbbp",
            "3dclp",
            "cvrp",
            "fjsp",
            "fssp",
            "jsp",
            "max_cut",
            "mis",
            "ossp",
            "rcpsp",
            "salbp",
            "tsp",
        )
        for problem in problems:
            with self.subTest(problem=problem):
                domain = importlib.import_module(f"src.problems.{problem}.domain_evaluator")
                instances = []
                for index in range(5):
                    instance = copy.deepcopy(synthetic_instance(problem))
                    instance["name"] = f"synthetic_{problem}_{index}"
                    instances.append(instance)
                synthetic_domain = SimpleNamespace(
                    load_instance_group=lambda _group, values=instances: copy.deepcopy(values),
                    describe_instance=domain.describe_instance,
                )
                first = split_domain_instances(synthetic_domain, "synthetic", seed=11)
                first_ids = {
                    name: [item["id"] for item in first.manifest[name]]
                    for name in ("train", "validation", "test")
                }
                groups = [set(first_ids[name]) for name in ("train", "validation", "test")]
                self.assertTrue(all(groups))
                self.assertEqual(
                    sum(map(len, groups)),
                    sum(len(first_ids[name]) for name in first_ids),
                )
                self.assertFalse(
                    groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]
                )


class InitializerTests(unittest.TestCase):
    def test_valid_llm_manifest_is_bounded_and_invalid_output_degrades(self):
        import tempfile

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            slots = Path(temporary) / "slots"
            for component in ("initialization|i", "operator|a", "operator|b", "operator|c"):
                category, name = component.split("|")
                directory = slots / category / name
                directory.mkdir(parents=True)
                (directory / "v1.py").write_text(
                    "def run(env_data, state, calc_makespan_fn):\n    return state\n",
                    encoding="utf-8",
                )
            config = {
                "initialization": {
                    "retries": 0,
                    "max_operator_count": {"fallback": 4, "min": 4, "max": 4, "llm_select": True},
                    "max_pipeline_length": {"fallback": 4, "min": 2, "max": 4, "llm_select": True},
                    "sampling_temperature": {
                        "fallback": 0.7,
                        "min": 0.1,
                        "max": 1.0,
                        "llm_select": True,
                    },
                }
            }

            class LLM:
                def __init__(self, answer):
                    self.answer = answer

                def chat(self, messages):
                    return self.answer, 0

            valid = LLM(
                """{"H":{"max_operator_count":4,"max_pipeline_length":3,"sampling_temperature":0.5},"operators":["initialization|i","operator|a"],"entry_nodes":["initialization|i"],"exit_nodes":["operator|a"],"edges":[["initialization|i","operator|a"]]}"""
            )
            initialized = PaperInitializer(
                slots,
                "prompt",
                ["initialization|i", "operator|a"],
                config,
                valid,
            ).initialize()
            self.assertFalse(initialized.degraded_initialization)
            self.assertEqual(initialized.max_pipeline_length, 3)
            self.assertEqual(initialized.sampling_temperature, 0.5)

            degraded = PaperInitializer(
                slots,
                "prompt",
                ["initialization|i", "operator|a"],
                config,
                LLM("not json"),
            ).initialize()
            self.assertTrue(degraded.degraded_initialization)
            self.assertEqual(
                degraded.graph.shortest_path(START, END),
                [START, "initialization|i", "operator|a", END],
            )


class PolicyUnitTests(unittest.TestCase):
    def test_constrained_log_prob_counts_only_generated_action_tokens(self):
        graph = DirectedOperatorGraph(
            {"initialization|a", "initialization|b"},
            {
                (START, "initialization|a"),
                (START, "initialization|b"),
                ("initialization|a", END),
                ("initialization|b", END),
            },
            {"initialization|a", "initialization|b"},
            {"initialization|a", "initialization|b"},
        )
        policy = PipelinePolicy.__new__(PipelinePolicy)
        policy.torch = torch
        policy.device = torch.device("cpu")
        policy._encode_prompt = MethodType(lambda self, context: [99] * 100, policy)
        token_map = {"initialization|a": (1,), "initialization|b": (2,), END: (3,)}
        policy._alias_tokens = MethodType(lambda self, action: token_map[action], policy)
        logits = torch.tensor([0.0, 2.0, 0.0, 0.0], requires_grad=True)
        policy._next_logits = MethodType(lambda self, ids: logits, policy)
        sample = policy.sample("large padded prompt", graph, 1, 1.0, greedy=True)
        self.assertEqual(sample.nodes, ["initialization|a"])
        expected = torch.log_softmax(logits[[1, 2]], dim=0)[0]
        self.assertAlmostEqual(sample.log_prob.item(), expected.item())
        self.assertFalse(sample.log_prob.requires_grad)
        replayed = policy._replay_sample_log_prob(sample)
        self.assertTrue(replayed.requires_grad)
        self.assertAlmostEqual(replayed.item(), expected.item())

    def test_update_detaches_reward_clips_and_steps_once(self):
        policy = PipelinePolicy.__new__(PipelinePolicy)
        policy.torch = torch
        policy.device = torch.device("cpu")
        policy.max_grad_norm = 1.0
        policy.model = torch.nn.Linear(1, 1, bias=False)

        class CountingOptimizer(torch.optim.SGD):
            def __init__(self, params):
                super().__init__(params, lr=0.1)
                self.steps = 0

            def step(self, closure=None):
                self.steps += 1
                return super().step(closure)

        policy.optimizer = CountingOptimizer(policy.model.parameters())
        log_prob = policy.model.weight.sum()
        candidate = CandidateResult(PipelineSample(["a"], log_prob=log_prob), normalized_reward=2.0)
        with mock.patch(
            "torch.nn.utils.clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_
        ) as clipped:
            stats = policy.update([candidate])
        self.assertIsInstance(stats, PolicyUpdateStats)
        self.assertTrue(math.isfinite(stats.loss))
        self.assertGreaterEqual(stats.grad_norm, 0.0)
        self.assertEqual(policy.optimizer.steps, 1)
        clipped.assert_called_once()
        # Logging must not deepcopy the non-leaf autograd tensor.
        logged = candidate.to_log_dict()
        self.assertNotIn("log_prob", logged["sample"])

    def test_update_streams_context_token_graphs_and_steps_once(self):
        policy = PipelinePolicy.__new__(PipelinePolicy)
        policy.torch = torch
        policy.device = torch.device("cpu")
        policy.max_grad_norm = 1.0
        policy.model = torch.nn.Linear(1, 1, bias=False)

        class CountingOptimizer(torch.optim.SGD):
            def __init__(self, params):
                super().__init__(params, lr=0.1)
                self.steps = 0

            def step(self, closure=None):
                self.steps += 1
                return super().step(closure)

        policy.optimizer = CountingOptimizer(policy.model.parameters())
        replayed = []

        def stream(self, sample):
            for _ in range(2):
                replayed.append(True)
                yield self.model.weight.sum()

        policy._iter_replay_sample_log_terms = MethodType(stream, policy)
        candidate = CandidateResult(
            PipelineSample(
                ["a"],
                policy_context="context",
                sampling_temperature=1.0,
                max_pipeline_length=1,
            ),
            normalized_reward=2.0,
        )
        stats = policy.update([candidate])
        self.assertTrue(math.isfinite(stats.loss))
        self.assertEqual(len(replayed), 2)
        self.assertEqual(policy.optimizer.steps, 1)


if __name__ == "__main__":
    unittest.main()
