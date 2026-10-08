from __future__ import annotations

import collections
import concurrent.futures
import importlib
import importlib.util
import math
import multiprocessing
import os
import sys
import time
import traceback
from typing import Any

from .contracts import (
    CandidateResult,
    PipelineSample,
    candidate_reward,
    reward_spec_from_domain,
    standardize_candidate_rewards,
)


class SolutionState:
    """Mutable solution state shared by every paper operator."""

    def __init__(self, sequence=None, makespan=float("inf")):
        self.sequence = sequence if sequence is not None else []
        self.makespan = makespan
        self.critical_path = []
        self.metadata = {
            "temp": 30.0,
            "no_improve": 0,
            "tabu_list": collections.deque(maxlen=80),
        }

    def copy(self):
        state = SolutionState(sequence=self.sequence[:], makespan=self.makespan)
        state.critical_path = self.critical_path[:]
        state.metadata = self.metadata.copy()
        if "tabu_list" in self.metadata:
            source = self.metadata["tabu_list"]
            state.metadata["tabu_list"] = collections.deque(source, maxlen=getattr(source, "maxlen", 80))
        return state


class PipelineExecutor:
    """Load operator implementations and execute one instantiated pipeline."""

    def __init__(self, slots_dir: str):
        self.slots_dir = slots_dir
        self.loaded_functions: dict[str, Any] = {}

    def _load_operator(self, category: str, name: str, version: str):
        key = f"{category}_{name}_{version}"
        if key in self.loaded_functions:
            return self.loaded_functions[key]
        path = os.path.join(self.slots_dir, category, name, f"{version}.py")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Operator implementation not found: {path}")
        spec = importlib.util.spec_from_file_location(key, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import operator implementation: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        run = getattr(module, "run", None)
        if not callable(run):
            raise ValueError(f"Operator implementation has no callable run(): {path}")
        self.loaded_functions[key] = run
        return run

    def execute(
        self,
        env_data: dict,
        pipeline: list[tuple[str, str, str]],
        domain_evaluator,
        max_iterations: int,
        initial_state: SolutionState | None = None,
        timeout_seconds: float = 80.0,
        diagnostics: dict | None = None,
    ):
        if not pipeline:
            return [], float("inf"), None, None
        diagnostics = diagnostics if diagnostics is not None else {}
        faults = diagnostics.setdefault("operator_faults", {})
        current_component = None
        current_implementation = None

        def record_fault(message: str):
            if current_implementation is None:
                return
            record = faults.setdefault(current_implementation, {"count": 0, "messages": []})
            record["count"] += 1
            if message not in record["messages"] and len(record["messages"]) < 3:
                record["messages"].append(message)

        try:
            started = time.monotonic()
            functions = []
            for category, name, version in pipeline:
                current_component = f"{category}|{name}"
                current_implementation = f"{current_component}|{version}"
                functions.append(self._load_operator(category, name, version))
            current_component = None
            current_implementation = None

            def evaluate(candidate, candidate_env=None):
                if candidate is env_data:
                    record_fault(
                        "calc_makespan_fn arguments were reversed; expected "
                        "calc_makespan_fn(candidate, env_data)"
                    )
                    candidate = candidate_env
                if getattr(domain_evaluator, "SEQUENCE_ONLY_OBJECTIVE", False):
                    return domain_evaluator.calc_makespan(candidate, env_data)
                if hasattr(candidate, "sequence"):
                    evaluation_state = candidate
                else:
                    # Trial sequences share the current state's auxiliary
                    # decisions (for example FJSP machine allocations).
                    # Keeping those decisions in the state avoids leaking them
                    # through process-global evaluator state between instances.
                    evaluation_state = state.copy()
                    evaluation_state.sequence = list(candidate)
                return domain_evaluator.calc_makespan(evaluation_state, env_data)

            def validate(state):
                if state is None:
                    raise TypeError("Operator returned None instead of SolutionState")
                if not hasattr(state, "sequence"):
                    raise TypeError("Operator returned an object without a sequence")
                # Pass the full state so domains with auxiliary decision variables
                # (for example FJSP machine allocations) validate the solution that
                # the operator actually produced instead of process-global fallback
                # state left behind by another instance.
                actual = float(domain_evaluator.calc_makespan(state, env_data))
                invalid = getattr(domain_evaluator, "INVALID_SCORE", None)
                if not math.isfinite(actual):
                    raise ValueError("Operator produced a non-finite objective")
                if invalid is not None and actual >= float(invalid):
                    raise ValueError(f"Operator produced an invalid solution (score={actual})")
                try:
                    stored = float(state.makespan)
                except (TypeError, ValueError):
                    stored = float("nan")
                if not math.isfinite(stored) or not math.isclose(
                    stored, actual, rel_tol=1e-9, abs_tol=1e-6
                ):
                    record_fault(f"state.makespan mismatch: stored={stored}, actual={actual}")
                    state.makespan = actual
                return state

            if initial_state is not None:
                state = initial_state.copy()
                loop_start = 1
            else:
                state = SolutionState()
                if pipeline[0][0].startswith("initialization"):
                    category, name, version = pipeline[0]
                    current_component = f"{category}|{name}"
                    current_implementation = f"{current_component}|{version}"
                    state = functions[0](env_data, state, evaluate)
                    loop_start = 1
                else:
                    loop_start = 0

            domain_evaluator.initialize_state(env_data, state)
            state = validate(state)
            best_sequence = list(state.sequence)
            best_value = float(state.makespan)
            loop = functions[loop_start:]

            for iteration in range(max_iterations):
                elapsed = time.monotonic() - started
                if elapsed > timeout_seconds:
                    diagnostics["timeout"] = {
                        "completed_iterations": iteration,
                        "max_iterations": max_iterations,
                        "elapsed_seconds": elapsed,
                        "timeout_seconds": timeout_seconds,
                    }
                    return best_sequence, best_value, None, None

                if state.metadata.get("force_reset", False):
                    if loop_start == 1:
                        category, name, version = pipeline[0]
                        current_component = f"{category}|{name}"
                        current_implementation = f"{current_component}|{version}"
                        state = functions[0](env_data, state, evaluate)
                    domain_evaluator.initialize_state(env_data, state)
                    state = validate(state)
                    state.metadata["force_reset"] = False

                state.metadata["prev_sequence"] = list(state.sequence)
                state.metadata["prev_makespan"] = state.makespan
                if "machine_alloc" in state.metadata:
                    state.metadata["prev_alloc"] = state.metadata["machine_alloc"].copy()
                for offset, function in enumerate(loop):
                    category, name, version = pipeline[offset + loop_start]
                    current_component = f"{category}|{name}"
                    current_implementation = f"{current_component}|{version}"
                    state = validate(function(env_data, state, evaluate))
                if state.makespan < best_value:
                    best_value = state.makespan
                    best_sequence = list(state.sequence)
            return best_sequence, best_value, None, None
        except Exception:
            return [], float("inf"), current_component, traceback.format_exc()


def execute_pipeline_worker(
    env_data,
    domain_evaluator,
    slots_dir,
    pipeline,
    pipeline_steps,
    initial_state=None,
    pipeline_timeout=80.0,
):
    """Process-safe entry point used by the N x k batch evaluator."""
    problem_dir = os.path.dirname(slots_dir)
    if problem_dir not in sys.path:
        sys.path.insert(0, problem_dir)
    if isinstance(domain_evaluator, str):
        domain_evaluator = importlib.import_module(domain_evaluator)
    diagnostics = {}
    started = time.time()
    sequence, score, crashed, error = PipelineExecutor(slots_dir).execute(
        env_data,
        pipeline,
        domain_evaluator,
        max_iterations=pipeline_steps,
        initial_state=initial_state,
        timeout_seconds=pipeline_timeout,
        diagnostics=diagnostics,
    )
    return (
        sequence,
        pipeline,
        score,
        crashed,
        error,
        time.time() - started,
        diagnostics.get("operator_faults", {}),
        diagnostics.get("timeout"),
    )


def _seeded_worker(seed: int, *args):
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except Exception:
        pass
    return execute_pipeline_worker(*args)


class RewardBatchEvaluator:
    def __init__(
        self,
        domain_evaluator,
        slots_dir: str,
        pipeline_steps: int,
        pipeline_timeout: float,
        failure_reward: float,
        reward_epsilon: float,
        max_workers: int = 4,
    ):
        self.domain_evaluator = domain_evaluator
        self.slots_dir = slots_dir
        self.pipeline_steps = int(pipeline_steps)
        self.pipeline_timeout = float(pipeline_timeout)
        self.failure_reward = float(failure_reward)
        self.reward_epsilon = float(reward_epsilon)
        self.max_workers = max(1, int(max_workers))

    def evaluate(
        self,
        samples: list[PipelineSample],
        instances: list[dict[str, Any]],
        standardize: bool = True,
        evaluation_seed: int = 0,
    ) -> list[CandidateResult]:
        results = [CandidateResult(sample=sample) for sample in samples]
        if not samples or not instances:
            raise ValueError("Evaluation requires candidates and benchmark instances")

        job_count = len(samples) * len(instances)
        max_workers = min(job_count, os.cpu_count() or 1, self.max_workers)
        completed_pairs: set[tuple[int, int]] = set()
        score_slots: list[list[float | None]] = [[None] * len(instances) for _ in samples]
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        batch_timed_out = False
        try:
            future_map = {}
            for candidate_index, sample in enumerate(samples):
                for instance_index, env_data in enumerate(instances):
                    # The same benchmark instance receives the same random
                    # stream across all candidates for fair comparison.
                    future = executor.submit(
                        _seeded_worker,
                        int(evaluation_seed) + instance_index,
                        env_data,
                        self.domain_evaluator.__name__,
                        self.slots_dir,
                        sample.implementations,
                        self.pipeline_steps,
                        None,
                        self.pipeline_timeout,
                    )
                    future_map[future] = (candidate_index, instance_index)
            waves = math.ceil(job_count / max_workers)
            try:
                for future in concurrent.futures.as_completed(
                    future_map,
                    timeout=self.pipeline_timeout * waves + 15.0,
                ):
                    candidate_index, instance_index = future_map[future]
                    completed_pairs.add((candidate_index, instance_index))
                    candidate = results[candidate_index]
                    try:
                        (
                            sequence,
                            _pipeline,
                            score,
                            crashed_component,
                            error_msg,
                            _exec_time,
                            operator_faults,
                            timeout_info,
                        ) = future.result()
                    except Exception as exc:
                        candidate.failed = True
                        candidate.failure_reason = f"instance_index={instance_index}: {exc!r}"
                        continue
                    if instance_index == 0:
                        candidate.representative_sequence = sequence
                    try:
                        numeric_score = float(score)
                    except (TypeError, ValueError, OverflowError):
                        numeric_score = float("nan")
                    if (
                        crashed_component is not None
                        or error_msg
                        or operator_faults
                        or timeout_info
                        or not math.isfinite(numeric_score)
                    ):
                        candidate.failed = True
                        candidate.crashed_component = crashed_component
                        if timeout_info:
                            candidate.timeout_info = {
                                **timeout_info,
                                "instance_index": instance_index,
                            }
                        reason = error_msg or (
                            "pipeline timeout" if timeout_info else "operator contract fault"
                        )
                        candidate.failure_reason = f"instance_index={instance_index}: {reason}"
                        candidate.operator_faults.update(operator_faults or {})
                    else:
                        score_slots[candidate_index][instance_index] = numeric_score
            except concurrent.futures.TimeoutError:
                batch_timed_out = True
                for future in future_map:
                    future.cancel()
        finally:
            if batch_timed_out:
                for process in list((getattr(executor, "_processes", None) or {}).values()):
                    if process.is_alive():
                        process.terminate()
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)

        for candidate_index, candidate in enumerate(results):
            expected = {
                (candidate_index, instance_index) for instance_index in range(len(instances))
            }
            missing_indices = [
                index for index, score in enumerate(score_slots[candidate_index]) if score is None
            ]
            candidate.scores = [
                float(score) for score in score_slots[candidate_index] if score is not None
            ]
            if not expected <= completed_pairs:
                candidate.failed = True
                unfinished = sorted(
                    instance_index
                    for pair_candidate, instance_index in expected - completed_pairs
                    if pair_candidate == candidate_index
                )
                candidate.failure_reason = (
                    f"batch evaluation timeout; unfinished_instance_indices={unfinished}"
                )
            elif missing_indices and not candidate.failure_reason:
                candidate.failed = True
                candidate.failure_reason = f"missing scores for instance_indices={missing_indices}"
            if candidate.failed or missing_indices:
                candidate.failed = True
                candidate.raw_reward = self.failure_reward
                continue
            specs = [
                reward_spec_from_domain(self.domain_evaluator, env_data) for env_data in instances
            ]
            candidate.raw_reward = candidate_reward(candidate.scores, specs, self.reward_epsilon)

        if standardize:
            standardize_candidate_rewards(results, self.reward_epsilon)
        return results
