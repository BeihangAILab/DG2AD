"""
FSSP domain evaluator — Flow Shop Scheduling Problem (Permutation).

Exposes the DGA2D domain protocol:
  load_instance_group, get_reward_spec, describe_instance, calc_makespan,
  get_initial_pipeline, initialize_state, get_prompt

Prompt text is read from prompts/fssp/domain_knowledge.txt.

FSSP encoding:
  - state.sequence: array of length N * M, each job j appears M times.
    The k-th occurrence of job j represents job j's operation on machine k.
  - No machine_selection needed (pure permutation problem).
"""

import os
import json
import numpy as np

from src.core.configuration import problem_data_dir

# ── Module-level config (set by Hydra before loading) ──
INVALID_SCORE = 999999999.0


def _get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _get_instances_path():
    private_fallback = os.path.join(_get_project_root(), "src", "problems", "fssp", "instances")
    return str(problem_data_dir("fssp", private_fallback) / "fssp_instances.json")


# ── Instance loading ──────────────────────────────────────────────


def _load_all_instances():
    """Load and parse fssp_instances.json into raw instance dicts."""
    path = _get_instances_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"FSSP instances file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return [e for e in entries if "processing_times" in e]


def _inst_to_env(inst):
    """Convert a raw FSSP JSON entry to the standard env_data format."""
    pt = np.array(inst["processing_times"], dtype=np.int32)
    num_machines, num_jobs = pt.shape
    return {
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "processing_times": pt,
        "total_ops": num_jobs * num_machines,
        "lower_bound": float(inst.get("lower_bound", 1)),
        "upper_bound": float(inst.get("upper_bound", 1)),
        "name": inst["name"],
        "group": inst.get("group", ""),
        "seed": inst.get("seed", 0),
    }


def load_instance_group(group):
    """Load every instance in the requested dataset group."""
    all_insts = _load_all_instances()
    group_insts = [e for e in all_insts if e.get("group") == group]
    return [_inst_to_env(inst) for inst in group_insts]


# ── Makespan computation (core) ───────────────────────────────────


def _calc_makespan_core(seq_arr, env_data):
    """
    Pure Python FSSP permutation flow shop schedule simulator.

    Args:
        seq_arr: np.ndarray of job IDs (dtype int32), length N * M.
                 The k-th occurrence of job j = operation on machine k.
        env_data: FSSP env dict with 'processing_times' (M × N).

    Returns:
        float makespan, or 999999999 if schedule is invalid.
    """
    pt = env_data["processing_times"]  # shape (M, N)
    num_machines, num_jobs = pt.shape
    total_ops = num_jobs * num_machines

    if len(seq_arr) != total_ops:
        return INVALID_SCORE
    if np.any(seq_arr < 0) or np.any(seq_arr >= num_jobs):
        return INVALID_SCORE
    if not np.all(np.bincount(seq_arr, minlength=num_jobs) == num_machines):
        return INVALID_SCORE

    # Track how many times each job has been seen -> current machine index
    job_op_idx = np.zeros(num_jobs, dtype=np.int32)
    # Machine availability timeline
    machine_avail = np.zeros(num_machines, dtype=np.float64)
    # Job completion time (last machine processed for this job)
    job_ready = np.zeros(num_jobs, dtype=np.float64)

    for pos in range(total_ops):
        job_id = int(seq_arr[pos])
        machine_id = int(job_op_idx[job_id])

        if machine_id >= num_machines:
            return INVALID_SCORE

        proc_time = float(pt[machine_id, job_id])

        # start = max(machine free, job ready after previous op)
        start_time = max(machine_avail[machine_id], job_ready[job_id])
        end_time = start_time + proc_time

        machine_avail[machine_id] = end_time
        job_ready[job_id] = end_time
        job_op_idx[job_id] += 1

    return float(max(machine_avail))


# ── Public evaluation interface ───────────────────────────────────


def calc_makespan(sequence_or_state, env_data):
    """
    Compute FSSP makespan.

    Polymorphic calling convention:
      - Engine: calc_makespan(sequence_list, env_data)
      - Slot functions: calc_makespan(state_obj, env_data)
        Extracts state.sequence automatically.

    Returns:
        float makespan (lower is better), or 999999999 for invalid.
    """
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
        seq_arr = raw.astype(np.int32)
    except (TypeError, ValueError, OverflowError):
        return INVALID_SCORE
    return _calc_makespan_core(seq_arr, env_data)


# ── State initialisation ──────────────────────────────────────────


def initialize_state(env_data, state):
    """
    Ensure state has a feasible FSSP solution if uninitialised.

    Uses a simple NEH-style construction: sort jobs by total processing
    time descending, then insert each job greedily.
    """
    if state.sequence is not None and len(state.sequence) > 0 and state.makespan < float("inf"):
        return

    num_jobs = env_data["num_jobs"]
    num_machines = env_data["num_machines"]
    pt = env_data["processing_times"]

    # Sort jobs by descending total processing time (LPT)
    total_pt = np.sum(pt, axis=0)
    job_order = list(np.argsort(-total_pt))

    def perm_to_seq(perm):
        return np.repeat(np.array(perm, dtype=np.int32), num_machines)

    def eval_perm(perm):
        seq = perm_to_seq(perm)
        return _calc_makespan_core(seq, env_data)

    # Greedy insertion
    partial = [job_order[0]]
    for i in range(1, num_jobs):
        job = job_order[i]
        best_mk = float("inf")
        best_pos = 0
        for pos in range(len(partial) + 1):
            candidate = partial[:pos] + [job] + partial[pos:]
            mk = eval_perm(candidate)
            if mk < best_mk:
                best_mk = mk
                best_pos = pos
        partial = partial[:best_pos] + [job] + partial[best_pos:]

    best_seq = perm_to_seq(partial)
    state.sequence = best_seq.copy()
    state.makespan = best_mk
    state.metadata = {}


# ── Pipeline configuration ────────────────────────────────────────


def get_initial_pipeline():
    """Return the default FSSP operator pipeline."""
    return [
        "initialization|init_fssp",
        "mutation|mutation_fssp",
        "local_search|ls_fssp",
        "acceptance|accept_fssp",
    ]


# ── Domain prompt ─────────────────────────────────────────────────


def get_prompt(root_dir=None):
    """
    Read the FSSP domain knowledge prompt from disk.

    Args:
        root_dir: Project root directory. If None, auto-detected relative
                  to this file (problems/fssp/ -> src/ -> root/).
    """
    if root_dir is None:
        root_dir = _get_project_root()
    prompt_path = os.path.join(root_dir, "prompts", "fssp", "domain_knowledge.txt")
    if not os.path.exists(prompt_path):
        return ""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_reward_spec(env_data):
    from src.core.contracts import default_reward_spec

    return default_reward_spec(env_data)


def describe_instance(env_data):
    from src.core.contracts import default_describe_instance

    return default_describe_instance(env_data)
