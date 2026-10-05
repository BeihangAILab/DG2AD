"""Regression checks for scoring a trial machine assignment through the executor."""
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.core.execution import PipelineExecutor, SolutionState
from src.problems.fjsp import domain_evaluator as domain
from tests.synthetic_domains import synthetic_instance

ROOT = Path(__file__).resolve().parents[1]


class TrialAllocationTests(unittest.TestCase):
    def test_state_copy_accepts_operator_list_tabu_history(self):
        state = SolutionState([0, 1])
        state.metadata["tabu_list"] = [("swap", 0, 1)]
        copied = state.copy()
        copied.metadata["tabu_list"].append(("machine", 0, 0, 1))
        self.assertEqual(list(state.metadata["tabu_list"]), [("swap", 0, 1)])
        self.assertEqual(len(copied.metadata["tabu_list"]), 2)

    def test_trial_score_uses_proposed_allocation_without_mutating_solution(self):
        env = synthetic_instance("fjsp")
        state = SolutionState([0, 1, 0, 1])
        state.metadata["machine_alloc"] = np.zeros((2, 2), dtype=np.int32)
        original = state.metadata["machine_alloc"].copy()
        trial = np.asarray([[0, 1], [1, 0]], dtype=np.int32)
        score = domain.evaluate_trial(state, state.sequence, trial, env, domain.calc_makespan)
        expected = state.copy()
        expected.metadata["machine_alloc"] = trial
        self.assertEqual(score, domain.calc_makespan(expected, env))
        self.assertNotEqual(score, domain.calc_makespan(state, env))
        np.testing.assert_array_equal(state.metadata["machine_alloc"], original)

    def test_acceptance_before_machine_search_restores_complete_snapshot(self):
        slots = ROOT / "src/problems/fjsp/slots"
        pipeline = [
            ("initialization", "init_alys_greedy", "v1"),
            ("coupled_perturbation", "ruins_and_recreate", "v1"),
            ("acceptance", "simulated_annealing", "v1"),
            ("intensification", "vns_descent", "v1"),
        ]
        for seed in range(20):
            with self.subTest(seed=seed):
                random.seed(seed)
                env = synthetic_instance("fjsp")
                env["op_counts"] = np.asarray([4, 4], dtype=np.int32)
                env["total_ops"] = 8
                env["max_ops"] = 4
                for key in ("processing_matrix", "machines_matrix", "times_matrix"):
                    env[key] = np.tile(env[key], (1, 2, 1))
                env["eligible_counts"] = np.full((2, 4), 2, dtype=np.int32)
                env["num_ops"] = env["op_counts"]
                env["max_operations"] = env["max_ops"]
                state = SolutionState([0] * 4 + [1] * 4)
                state.metadata["machine_alloc"] = np.zeros((2, 4), dtype=np.int32)
                state.metadata["temp"] = 0.1
                state.makespan = domain.calc_makespan(state, env)
                diagnostics = {}
                with mock.patch.dict(sys.modules, {"domain_evaluator": domain}):
                    result = PipelineExecutor(str(slots)).execute(
                        env, pipeline, domain, 100, initial_state=state, diagnostics=diagnostics,
                    )
                self.assertIsNone(result[3], diagnostics)
                self.assertFalse(diagnostics.get("operator_faults"), diagnostics)

    def test_search_operators_commit_the_score_of_their_actual_solution(self):
        operators = [
            ("intensification", "tabu_search_lite"),
            ("intensification", "vns_descent"),
            ("machine_ls", "ls_machine_greedy_balance"),
            ("machine_ls", "ls_machine_idle_fill"),
            ("critical_path_ls", "ls_critical_machine_reassign"),
        ]
        slots = ROOT / "src/problems/fjsp/slots"
        for category, name in operators:
            for seed in range(5):
                with self.subTest(operator=name, seed=seed):
                    random.seed(seed)
                    env = synthetic_instance("fjsp")
                    env["num_ops"] = env["op_counts"]
                    state = SolutionState([0, 0, 1, 1])
                    state.metadata["machine_alloc"] = np.zeros((2, 2), dtype=np.int32)
                    state.makespan = domain.calc_makespan(state, env)
                    starting_score = state.makespan
                    diagnostics = {}
                    with mock.patch.dict(sys.modules, {"domain_evaluator": domain}):
                        result = PipelineExecutor(str(slots)).execute(
                            env, [("initialization", "init_random", "v1"), (category, name, "v1")],
                            domain, 50, initial_state=state, diagnostics=diagnostics,
                        )
                    self.assertIsNone(result[3], diagnostics)
                    self.assertFalse(diagnostics.get("operator_faults"), diagnostics)
                    self.assertTrue(np.isfinite(result[1]))
                    if name in {"vns_descent", "ls_machine_greedy_balance"}:
                        self.assertLess(result[1], starting_score)


if __name__ == "__main__":
    unittest.main()
