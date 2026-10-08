import importlib
import tempfile
import unittest
from pathlib import Path

from src.core.operators import (
    validate_candidate_code,
    validate_candidate_runtime,
)
from src.core.execution import PipelineExecutor
from tests.synthetic_domains import synthetic_instance


ROOT = Path(__file__).resolve().parents[1]


class TinyDomain:
    INVALID_SCORE = 999.0

    @staticmethod
    def calc_makespan(sequence_or_state, env_data):
        sequence = getattr(sequence_or_state, "sequence", sequence_or_state)
        return float(sum(sequence)) if len(sequence) == 2 else TinyDomain.INVALID_SCORE

    @staticmethod
    def initialize_state(env_data, state):
        if not state.sequence:
            state.sequence = [0, 1]
            state.makespan = 1.0


class RuntimeMigrationTests(unittest.TestCase):
    def make_operator(self, root, category, name, code):
        directory = Path(root) / category / name
        directory.mkdir(parents=True)
        (directory / "v1.py").write_text(code, encoding="utf-8")

    def test_executor_preserves_fault_crash_and_timeout_diagnostics(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            self.make_operator(
                temporary,
                "initialization",
                "init",
                "def run(env_data, state, calc_makespan_fn):\n"
                "    state.sequence = [0, 1]\n"
                "    state.makespan = 7.0\n"
                "    return state\n",
            )
            self.make_operator(
                temporary,
                "search",
                "boom",
                "def run(env_data, state, calc_makespan_fn):\n"
                "    raise RuntimeError('operator exploded')\n",
            )
            self.make_operator(
                temporary,
                "search",
                "slow",
                "import time\n"
                "def run(env_data, state, calc_makespan_fn):\n"
                "    time.sleep(0.02)\n"
                "    return state\n",
            )
            executor = PipelineExecutor(temporary)
            diagnostics = {}
            _, _, crashed, error = executor.execute(
                {},
                [("initialization", "init", "v1"), ("search", "boom", "v1")],
                TinyDomain,
                1,
                diagnostics=diagnostics,
            )
            self.assertEqual(crashed, "search|boom")
            self.assertIn("operator exploded", error)
            self.assertIn("initialization|init|v1", diagnostics["operator_faults"])

            diagnostics = {}
            sequence, score, crashed, error = executor.execute(
                {},
                [("initialization", "init", "v1"), ("search", "slow", "v1")],
                TinyDomain,
                2,
                timeout_seconds=0.001,
                diagnostics=diagnostics,
            )
            self.assertEqual(sequence, [0, 1])
            self.assertEqual(score, 1.0)
            self.assertIsNone(crashed)
            self.assertIsNone(error)
            self.assertIn(diagnostics["timeout"]["completed_iterations"], (0, 1))
            self.assertEqual(diagnostics["timeout"]["max_iterations"], 2)

    def test_generated_code_ast_signature_and_isolated_runtime_validation(self):
        valid = (
            "def run(env_data, state, calc_makespan_fn):\n"
            "    state.sequence = list(range(env_data['num_nodes']))\n"
            "    state.makespan = calc_makespan_fn(state.sequence, env_data)\n"
            "    return state\n"
        )
        self.assertEqual(validate_candidate_code(valid), (True, None))
        wrong = "def run(state, env_data, score):\n    return state\n"
        ok, message = validate_candidate_code(wrong)
        self.assertFalse(ok)
        self.assertIn("signature", message)

        evaluator = importlib.import_module("src.problems.tsp.domain_evaluator")
        env = synthetic_instance("tsp")
        ok, message = validate_candidate_runtime(
            valid,
            "initialization",
            env,
            evaluator.__name__,
            str(ROOT / "src" / "problems" / "tsp"),
            timeout_seconds=20.0,
        )
        self.assertTrue(ok, message)

    def test_runtime_validation_rejects_invalid_solution_and_timeout(self):
        evaluator = importlib.import_module("src.problems.tsp.domain_evaluator")
        env = synthetic_instance("tsp")
        cases = [
            ("def run(env_data, state, calc_makespan_fn):\n"
             "    state.sequence = []\n    return state\n", 20.0, "invalid solution"),
            ("def run(env_data, state, calc_makespan_fn):\n"
             "    while True: pass\n", 2.0, "exceeded"),
        ]
        for code, timeout, expected in cases:
            with self.subTest(expected=expected):
                ok, message = validate_candidate_runtime(
                    code, "initialization", env, evaluator.__name__,
                    str(ROOT / "src/problems/tsp"), timeout_seconds=timeout,
                )
                self.assertFalse(ok)
                self.assertIn(expected, message)


if __name__ == "__main__":
    unittest.main()
