"""
OSSP domain evaluator — Open Shop Scheduling Problem.

Exposes the DGA2D domain protocol:
  load_instance_group, get_reward_spec, describe_instance, calc_makespan,
  get_initial_pipeline, initialize_state, get_prompt

Encoding: permutation of operation IDs.
  op_id = job * M + machine,  range [0, N*M - 1]

The decoder processes operations in sequence order and starts each
operation as soon as both its job and machine are free (greedy list
scheduling). No preemption.

Objective: makespan (max completion time across all jobs/machines).
This is a MINIMISATION problem.
"""

import json
import os
import random
import numpy as np

from src.core.configuration import problem_data_dir

# ── Module-level config (set by Hydra before loading) ──
INVALID_SCORE = 999999999.0


def _get_instances_path():
    """Return the absolute path to the OSSP instances JSON."""
    return str(problem_data_dir("ossp", os.path.dirname(__file__)) / "ossp_instances.json")


def _load_group(group_str):
    """Load all instances for a given group key from the JSON database."""
    with open(_get_instances_path(), "r", encoding="utf-8") as f:
        all_data = json.load(f)

    if group_str in all_data:
        return all_data[group_str]

    # Try prefix matching (e.g. "tai10" → "tai10_10.txt")
    for key in all_data:
        if key.startswith(group_str):
            return all_data[key]

    return []


def _inst_to_env(inst):
    """Convert a raw OSSP instance dict to the standard env_data format."""
    times = np.array(inst["ossp_times"], dtype=np.int32)
    n_jobs, n_machines = times.shape
    return {
        "name": f"ossp_{n_jobs}x{n_machines}",
        "num_jobs": n_jobs,
        "num_machines": n_machines,
        "total_ops": n_jobs * n_machines,
        "ossp_times": times,
        "upper_bound": float(inst.get("upper_bound", 999999)),
        "lower_bound": float(inst.get("lower_bound", 0)),
    }


# ── Data loading (DGA2D domain protocol) ────────────────────────────


def load_instance_group(group):
    """Load every instance in the requested dataset group."""
    instances = _load_group(group)
    return [_inst_to_env(inst) for inst in instances]


# ── Solution evaluation ─────────────────────────────────────────────


def calc_makespan(sequence_or_state, env_data):
    """
    Compute OSSP makespan via greedy list scheduling.

    Processes operations in the given sequence order. Each operation
    (job, machine) pair starts at max(job_free_time, machine_free_time).

    Validates that the sequence contains exactly one of each operation.
    """
    n_jobs = env_data["num_jobs"]
    n_machines = env_data["num_machines"]
    times = env_data["ossp_times"]
    total_ops = n_jobs * n_machines

    # Polymorphic: handle SolutionState and raw sequence
    if hasattr(sequence_or_state, "sequence"):
        seq = sequence_or_state.sequence
    else:
        seq = sequence_or_state

    if seq is None:
        return INVALID_SCORE
    try:
        raw = np.asarray(seq).ravel()
        if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
            return INVALID_SCORE
        seq = raw.astype(np.int32)
    except (TypeError, ValueError, OverflowError):
        return INVALID_SCORE

    if len(seq) != total_ops:
        return INVALID_SCORE

    # Validate: each operation appears exactly once
    if seq.min() < 0 or seq.max() >= total_ops or len(set(seq)) != total_ops:
        return INVALID_SCORE

    job_avail = np.zeros(n_jobs, dtype=np.float64)
    mach_avail = np.zeros(n_machines, dtype=np.float64)

    for op in seq:
        job_id = op // n_machines
        mach_id = op % n_machines
        p_time = float(times[job_id, mach_id])
        start = max(job_avail[job_id], mach_avail[mach_id])
        end = start + p_time
        job_avail[job_id] = end
        mach_avail[mach_id] = end

    return float(max(job_avail))


# ── State initialisation ─────────────────────────────────────────────


def initialize_state(env_data, state):
    """
    Ensure state has a feasible OSSP solution.

    Creates a random permutation of all N*M operations.
    """
    if state.sequence is not None and len(state.sequence) > 0 and state.makespan < float("inf"):
        return

    total_ops = env_data["total_ops"]
    seq = list(range(total_ops))
    random.shuffle(seq)
    state.sequence = np.array(seq, dtype=np.int32)
    state.makespan = calc_makespan(state, env_data)


# ── Pipeline configuration ───────────────────────────────────────────


def get_initial_pipeline():
    """Return the default OSSP operator pipeline."""
    return [
        "initialization|init_random",
        "perturbation|mut_swap",
        "local_search|greedy_insertion",
        "acceptance|simulated_annealing",
    ]


# ── Domain prompt ────────────────────────────────────────────────────


def get_prompt(root_dir=None):
    """
    Read the OSSP domain knowledge prompt from disk.

    Args:
        root_dir: Project root directory. If None, auto-detected relative
                  to this file (problems/ossp/ -> src/ -> root/).
    """
    if root_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    prompt_path = os.path.join(root_dir, "prompts", "ossp", "domain_knowledge.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_reward_spec(env_data):
    from src.core.contracts import default_reward_spec

    return default_reward_spec(env_data)


def describe_instance(env_data):
    from src.core.contracts import default_describe_instance

    return default_describe_instance(env_data)
