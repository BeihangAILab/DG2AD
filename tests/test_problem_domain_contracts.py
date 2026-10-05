import ast
import importlib
import importlib.util
import random
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.core.contracts import (
    REQUIRED_DOMAIN_CALLABLES,
    validate_domain_evaluator,
)
from src.core.execution import SolutionState
from tests.synthetic_domains import synthetic_instance


ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = (
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


def load_slot(path):
    spec = importlib.util.spec_from_file_location("test_" + "_".join(path.parts[-4:-1]), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_first_instance(problem, evaluator):
    return synthetic_instance(problem)


class VersionPoolTests(unittest.TestCase):
    def test_every_executable_slot_has_v1_baseline(self):
        for problem in PROBLEMS:
            slots = ROOT / "src" / "problems" / problem / "slots"
            for directory in slots.rglob("*"):
                if not directory.is_dir() or directory.name == "__pycache__":
                    continue
                versions = sorted(path.name for path in directory.glob("v*.py"))
                if versions:
                    self.assertIn("v1.py", versions, str(directory))

    def test_all_v1_signatures_and_callback_order(self):
        files = sorted((ROOT / "src" / "problems").glob("*/slots/**/v1.py"))
        self.assertEqual(len(files), 120)
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            run = next(
                (
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "run"
                ),
                None,
            )
            self.assertIsNotNone(run, str(path))
            self.assertEqual(
                [arg.arg for arg in run.args.args],
                ["env_data", "state", "calc_makespan_fn"],
                str(path),
            )
            for node in ast.walk(run):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                if node.func.id == "calc_makespan_fn" and node.args:
                    self.assertFalse(
                        isinstance(node.args[0], ast.Name) and node.args[0].id == "env_data",
                        f"reversed objective callback in {path}",
                    )

    def test_domain_prompts_publish_operator_contract(self):
        for problem in PROBLEMS:
            text = (ROOT / "prompts" / problem / "domain_knowledge.txt").read_text(encoding="utf-8")
            if problem == "cvrp":
                self.assertIn("[OBJECTIVE CALLBACK CONTRACT]", text)
                self.assertIn("calc_makespan_fn(candidate_sequence, env_data)", text)
            else:
                self.assertIn("[OPERATOR CONTRACT]", text, problem)
                self.assertIn("calc_makespan_fn(candidate_or_state, env_data)", text, problem)

    def test_sensitive_domain_prompts_match_runtime_contracts(self):
        prompt_root = ROOT / "prompts"
        clp = (prompt_root / "3dclp/domain_knowledge.txt").read_text(encoding="utf-8")
        self.assertIn("orientation_allowed", clp)
        fjsp = (prompt_root / "fjsp/domain_knowledge.txt").read_text(encoding="utf-8")
        self.assertIn("ACTUAL machine ID", fjsp)
        self.assertIn("processing_matrix", fjsp)
        mis = (prompt_root / "mis/domain_knowledge.txt").read_text(encoding="utf-8")
        self.assertIn("INVALID_SCORE", mis)
        for problem in PROBLEMS:
            text = (prompt_root / problem / "domain_knowledge.txt").read_text(encoding="utf-8")
            self.assertNotIn("Meta-graph", text)


class EvaluatorContractTests(unittest.TestCase):
    def test_paper_rl_protocol_is_available_in_all_domains(self):
        for problem in PROBLEMS:
            with self.subTest(problem=problem):
                evaluator = importlib.import_module(f"src.problems.{problem}.domain_evaluator")
                validate_domain_evaluator(evaluator)
                for method in REQUIRED_DOMAIN_CALLABLES:
                    self.assertTrue(callable(getattr(evaluator, method, None)))
                self.assertFalse(hasattr(evaluator, "load_data"))
                self.assertFalse(hasattr(evaluator, "load_test_group"))
                env = load_first_instance(problem, evaluator)
                spec = evaluator.get_reward_spec(env).validate()
                self.assertIn(spec.direction, ("min", "max"))
                description = evaluator.describe_instance(env)
                self.assertIsInstance(description, str)
                self.assertNotIn("[[", description)

    def test_sequence_and_state_scores_match(self):
        for problem in PROBLEMS:
            with self.subTest(problem=problem):
                evaluator = importlib.import_module(f"src.problems.{problem}.domain_evaluator")
                self.assertTrue(hasattr(evaluator, "INVALID_SCORE"))
                env = load_first_instance(problem, evaluator)
                state = SolutionState()
                evaluator.initialize_state(env, state)
                state_score = evaluator.calc_makespan(state, env)
                raw_score = evaluator.calc_makespan(state.sequence, env)
                self.assertTrue(np.isfinite(state_score))
                self.assertLess(state_score, evaluator.INVALID_SCORE)
                self.assertAlmostEqual(state_score, raw_score, places=6)
                self.assertAlmostEqual(float(state.makespan), state_score, places=6)

    def test_invalid_encodings_use_invalid_score(self):
        for problem in PROBLEMS:
            with self.subTest(problem=problem):
                evaluator = importlib.import_module(f"src.problems.{problem}.domain_evaluator")
                env = load_first_instance(problem, evaluator)
                state = SolutionState()
                evaluator.initialize_state(env, state)
                sequence = list(np.asarray(state.sequence).ravel())
                invalid = sequence[:-1]
                self.assertGreaterEqual(
                    evaluator.calc_makespan(invalid, env), evaluator.INVALID_SCORE
                )

    def test_max_cut_synthetic_reference_is_consistent(self):
        evaluator = importlib.import_module("src.problems.max_cut.domain_evaluator")
        env = synthetic_instance("max_cut")
        self.assertEqual(env["best_known_cut"], 2.0)
        self.assertEqual(env["upper_bound"], -2.0)
        self.assertEqual(evaluator.calc_makespan([0, 1, 0], env), -2.0)


class FilteredDatasetGroupTests(unittest.TestCase):
    def test_max_cut_le2000_group_excludes_exact_large_instances(self):
        evaluator = importlib.import_module("src.problems.max_cut.domain_evaluator")
        filenames = [f"G{index}.txt" for index in range(1, 42)]
        filenames.extend(
            ["G50.txt", "G55.txt", "G60.txt", "G65.txt", "G70.txt", "G72.txt", "G81.txt"]
        )
        with (
            mock.patch.object(evaluator, "_get_instances_dir", return_value="unused"),
            mock.patch.object(evaluator.os, "listdir", return_value=filenames),
            mock.patch.object(
                evaluator,
                "load_maxcut_data",
                return_value={"num_nodes": 1, "num_edges": 0},
            ),
            mock.patch.object(evaluator, "_get_bks", return_value=1.0),
        ):
            loaded = evaluator.load_instance_group("G-set_le2000")
        self.assertEqual(len(loaded), 41)
        self.assertFalse(
            {item["name"][:-4] for item in loaded} & evaluator.GSET_LE2000_EXCLUDED
        )

    def test_mis_combined_group_deduplicates_case_insensitively(self):
        evaluator = importlib.import_module("src.problems.mis.domain_evaluator")
        valid = [f"instance_{index:03d}.clq" for index in range(112)]
        groups = {
            "BHOSLIB": valid[:50] + ["C4000.5.clq"],
            "DIMACS_all": valid[50:90] + ["INSTANCE_000.clq", "MANN_a81.clq"],
            "DIMACS_subset": valid[90:] + ["Instance_050.clq", "keller6.clq"],
        }
        original = evaluator._list_instances_in_group

        def listed(group):
            if group == evaluator.MIS_LE2000_GROUP:
                return original(group)
            return [str(Path(group) / name) for name in groups[group]]

        with mock.patch.object(evaluator, "_list_instances_in_group", side_effect=listed):
            paths = original(evaluator.MIS_LE2000_GROUP)
        stems = [Path(path).stem.casefold() for path in paths]
        self.assertEqual(len(stems), 112)
        self.assertEqual(stems, sorted(set(stems)))


class OperatorBehaviorTests(unittest.TestCase):
    def test_acceptance_rejects_worse_candidate_and_restores_parent(self):
        cases = (
            ("3dbbp", "acceptance/simulated_annealing"),
            ("3dclp", "acceptance/simulated_annealing"),
            ("3dclp", "acceptance/late_acceptance"),
            ("fjsp", "acceptance/simulated_annealing"),
            ("fssp", "acceptance/accept_fssp"),
            ("jsp", "acceptance/accept_hill_climbing"),
            ("jsp", "acceptance/simulated_annealing"),
            ("max_cut", "intensification/accept_maxcut"),
            ("mis", "acceptance/simulated_annealing"),
            ("ossp", "acceptance/simulated_annealing"),
            ("rcpsp", "acceptance/simulated_annealing"),
            ("salbp", "acceptance/simulated_annealing"),
            ("tsp", "acceptance/simulated_annealing"),
        )
        for problem, node in cases:
            with self.subTest(problem=problem, node=node):
                evaluator = importlib.import_module(f"src.problems.{problem}.domain_evaluator")
                sys.modules["domain_evaluator"] = evaluator
                module = load_slot(ROOT / "src" / "problems" / problem / "slots" / node / "v1.py")
                state = SolutionState([1, 0], 100.0)
                state.metadata.update(
                    {
                        "prev_sequence": [0, 1],
                        "prev_makespan": 5.0,
                        "best_score": 5.0,
                        "best_seq": np.array([0, 1], dtype=np.int32),
                        "best_makespan": 5.0,
                        "best_sequence": np.array([0, 1], dtype=np.int32),
                        "record": 5.0,
                        "deviation": 1.0,
                    }
                )
                if problem == "fjsp":
                    current_alloc = np.array([[1]], dtype=np.int32)
                    previous_alloc = np.array([[0]], dtype=np.int32)
                    state.metadata.update(
                        {
                            "machine_alloc": current_alloc,
                            "prev_alloc": previous_alloc,
                            "g_best_ms": 5.0,
                            "g_best_seq": [0, 1],
                            "g_best_alloc": previous_alloc,
                        }
                    )
                with (
                    mock.patch("random.random", return_value=1.0),
                    mock.patch("numpy.random.random", return_value=1.0),
                ):
                    module.run({}, state, lambda candidate, env: state.makespan)
                self.assertEqual(float(state.makespan), 5.0)
                self.assertEqual(list(state.sequence), [0, 1])
                if problem == "fjsp":
                    self.assertTrue(
                        np.array_equal(
                            state.metadata["machine_alloc"],
                            state.metadata["prev_alloc"],
                        )
                    )

    def test_local_search_commits_sequence_with_score(self):
        cases = (
            (
                "rcpsp",
                "local_search/ls_swap",
                {"num_activities": 3, "adj_matrix": np.zeros((3, 3), dtype=np.int32)},
            ),
            (
                "salbp",
                "local_search/ls_swap",
                {"num_tasks": 3, "adj_matrix": np.zeros((3, 3), dtype=np.int32)},
            ),
            ("ossp", "local_search/ls_2opt", {"total_ops": 3}),
        )
        for problem, node, env in cases:
            with self.subTest(problem=problem):
                module = load_slot(ROOT / "src" / "problems" / problem / "slots" / node / "v1.py")
                original = np.array([0, 1, 2], dtype=np.int32)
                state = SolutionState(original.copy(), 1.0)

                def score(candidate, _env):
                    seq = np.asarray(
                        candidate.sequence if hasattr(candidate, "sequence") else candidate
                    )
                    return 1.0 if np.array_equal(seq, original) else 0.0

                module.run(env, state, score)
                self.assertFalse(np.array_equal(state.sequence, original))
                self.assertEqual(state.makespan, score(state.sequence, env))

    def test_tsp_perturbation_accepts_legal_non_improvement(self):
        module = load_slot(ROOT / "src/problems/tsp/slots/perturbation/mut_swap/v1.py")
        state = SolutionState(np.arange(6, dtype=np.int32), 10.0)
        module.run({"num_nodes": 6}, state, lambda candidate, env: 20.0)
        self.assertEqual(sorted(state.sequence.tolist()), list(range(6)))
        self.assertFalse(np.array_equal(state.sequence, np.arange(6)))
        self.assertEqual(state.makespan, 20.0)

    def test_max_cut_mutation_forces_one_flip(self):
        module = load_slot(ROOT / "src/problems/max_cut/slots/perturbation/mutation_maxcut/v1.py")
        state = SolutionState([0, 0, 0], 0.0)
        with (
            mock.patch.object(random, "random", return_value=1.0),
            mock.patch.object(random, "randrange", return_value=0),
        ):
            module.run({"num_nodes": 3}, state, lambda candidate, env: -1.0)
        self.assertEqual(state.sequence, [1, 0, 0])
        self.assertEqual(state.makespan, -1.0)

    def test_fjsp_machine_allocation_changes_score_and_is_thread_local(self):
        evaluator = importlib.import_module("src.problems.fjsp.domain_evaluator")
        env = {
            "num_jobs": 1,
            "num_machines": 2,
            "op_counts": np.array([1], dtype=np.int32),
            "eligible_counts": np.array([[2]], dtype=np.int32),
            "machines_matrix": np.array([[[0, 1]]], dtype=np.int32),
            "times_matrix": np.array([[[1, 10]]], dtype=np.int32),
            "processing_matrix": np.array([[[1, 10]]], dtype=np.int32),
            "max_ops": 1,
            "total_ops": 1,
        }
        fast = SolutionState([0], 1.0)
        fast.metadata["machine_alloc"] = np.array([[0]], dtype=np.int32)
        slow = SolutionState([0], 10.0)
        slow.metadata["machine_alloc"] = np.array([[1]], dtype=np.int32)
        self.assertEqual(evaluator.calc_makespan(fast, env), 1.0)
        self.assertEqual(evaluator.calc_makespan(slow, env), 10.0)

        results = [None, None]

        def evaluate(index, allocation):
            evaluator.set_active_machine_allocation(np.array([[allocation]], dtype=np.int32))
            results[index] = evaluator.calc_makespan([0], env)

        threads = [
            threading.Thread(target=evaluate, args=(0, 0)),
            threading.Thread(target=evaluate, args=(1, 1)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(results, [1.0, 10.0])


if __name__ == "__main__":
    unittest.main()
