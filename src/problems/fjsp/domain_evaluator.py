"""
FJSP domain evaluator — Flexible Job-shop Scheduling Problem.

Exposes the DGA2D domain protocol:
  load_instance_group, get_reward_spec, describe_instance, calc_makespan,
  get_initial_pipeline, initialize_state, get_prompt

Prompt text is read from prompts/fjsp/domain_knowledge.txt.

FJSP-specific notes:
  - Dual-chromosome encoding: OS (operation sequence) + MS (machine selection).
  - calc_makespan is polymorphic: accepts (sequence, env_data) from the engine
    OR (state, env_data) from slot functions — detected automatically.
  - Machine selection is bound via a module-level variable when engine calls
    use plain sequences (no state object).
"""

import os
import json
import random
import threading
import numpy as np

from src.core.configuration import problem_data_dir
from src.problems.fjsp.schedule_kernel import simulate_schedule

# ── Module-level config (set by Hydra before loading) ──
INVALID_SCORE = 999999999.0

# ── Module-level binding for machine selection ──
# When calc_makespan is called with a bare sequence (from engine),
# the current machine selection must be bound beforehand.
_allocation_context = threading.local()


def set_active_machine_allocation(allocation):
    """Bind actual machine IDs for subsequent bare-sequence evaluation."""
    if allocation is None:
        _allocation_context.value = None
        return None
    bound = np.ascontiguousarray(np.asarray(allocation, dtype=np.int32))
    _allocation_context.value = bound
    return bound


def get_active_machine_allocation():
    return getattr(_allocation_context, "value", None)


def evaluate_trial(state, sequence, allocation, env_data, calc_makespan_fn):
    """Evaluate an OS/MS trial together without changing the current solution."""
    trial = state.copy()
    trial.sequence = list(sequence)
    trial.metadata["machine_alloc"] = np.array(allocation, dtype=np.int32, copy=True)
    return calc_makespan_fn(trial, env_data)


def _get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _get_instances_dir():
    private_fallback = os.path.join(_get_project_root(), "src", "problems", "fjsp", "instances")
    return str(problem_data_dir("fjsp", private_fallback))


def _load_bounds_index():
    """Load instances.json which contains known optima and bounds."""
    bounds_path = os.path.join(_get_instances_dir(), "instances.json")
    if not os.path.exists(bounds_path):
        return {}
    with open(bounds_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    idx = {}
    for e in entries:
        name = e["name"].lower()
        if e.get("optimum") is not None:
            idx[name] = {"lb": float(e["optimum"]), "ub": float(e["optimum"])}
        elif "bounds" in e:
            idx[name] = {
                "lb": float(e["bounds"]["lower"]),
                "ub": float(e["bounds"]["upper"]),
            }
        else:
            idx[name] = {"lb": 1.0, "ub": 999999.0}
    return idx


# ── Data loading ──────────────────────────────────────────────────


def parse_fjsp_txt(filepath):
    """
    Parse a standard FJSP .txt file (Brandimarte / Hurink format).

    Format:
      Line 1: <num_jobs> <num_machines> [avg_machines_per_op]
      Per job: <num_ops> [<num_eligible> <m1> <t1> <m2> <t2> ...] x num_ops

    Machine IDs are 1-based in file, normalised to 0-based in output.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FJSP instance not found: {filepath}")

    with open(filepath, "r") as f:
        lines = f.readlines()

    first = lines[0].strip().split()
    num_jobs = int(first[0])
    num_machines = int(first[1])

    # Flatten all tokens after the header line
    tokens = []
    for line in lines[1:]:
        tokens.extend(line.strip().split())

    jobs_data = []
    max_ops = 0
    max_eligible = 0
    min_machine_id = 9999
    idx = 0

    for j in range(num_jobs):
        if idx >= len(tokens):
            break
        num_ops = int(tokens[idx])
        idx += 1
        if num_ops > max_ops:
            max_ops = num_ops
        ops_data = []
        for o in range(num_ops):
            num_eligible = int(tokens[idx])
            idx += 1
            if num_eligible > max_eligible:
                max_eligible = num_eligible
            op_machines = []
            for _ in range(num_eligible):
                m_id = int(tokens[idx])
                t = int(tokens[idx + 1])
                if m_id < min_machine_id:
                    min_machine_id = m_id
                op_machines.append((m_id, t))
                idx += 2
            ops_data.append(op_machines)
        jobs_data.append(ops_data)

    offset = min_machine_id

    op_counts = np.zeros(num_jobs, dtype=np.int32)
    eligible_counts = np.zeros((num_jobs, max_ops), dtype=np.int32)
    machines_matrix = np.full((num_jobs, max_ops, max_eligible), -1, dtype=np.int32)
    times_matrix = np.full((num_jobs, max_ops, max_eligible), -1, dtype=np.int32)

    for j, ops_data in enumerate(jobs_data):
        op_counts[j] = len(ops_data)
        for o, op_machines in enumerate(ops_data):
            eligible_counts[j, o] = len(op_machines)
            for e, (m_id, t) in enumerate(op_machines):
                machines_matrix[j, o, e] = m_id - offset
                times_matrix[j, o, e] = t

    total_ops = int(sum(op_counts))

    return {
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "op_counts": op_counts,
        "eligible_counts": eligible_counts,
        "machines_matrix": machines_matrix,
        "times_matrix": times_matrix,
        "max_ops": max_ops,
        "total_ops": total_ops,
    }


def _inst_to_env(raw, inst_name, bounds_index):
    """Convert a raw parsed instance dict to the standard env_data format."""
    name_lower = inst_name.lower()
    bounds = bounds_index.get(name_lower, {"lb": 1.0, "ub": 999999.0})
    processing_matrix = np.full(
        (raw["num_jobs"], raw["max_ops"], raw["num_machines"]), -1, dtype=np.int32
    )
    for i in range(raw["num_jobs"]):
        for j in range(raw["max_ops"]):
            for k in range(raw["machines_matrix"].shape[2]):
                m = raw["machines_matrix"][i, j, k]
                if m != -1:
                    processing_matrix[i, j, m] = raw["times_matrix"][i, j, k]

    return {
        "num_jobs": raw["num_jobs"],
        "num_machines": raw["num_machines"],
        "op_counts": raw["op_counts"],
        "num_ops": raw["op_counts"],
        "eligible_counts": raw["eligible_counts"],
        "machines_matrix": raw["machines_matrix"],
        "times_matrix": raw["times_matrix"],
        "processing_matrix": processing_matrix,
        "max_ops": raw["max_ops"],
        "max_operations": raw["max_ops"],
        "total_ops": raw["total_ops"],
        "lower_bound": bounds["lb"],
        "upper_bound": bounds["ub"],
        "name": inst_name,
    }


def _list_instances_in_group(group):
    """List all .txt instance files in a group directory."""
    group_dir = os.path.join(_get_instances_dir(), group)
    if not os.path.isdir(group_dir):
        return []
    files = sorted([f for f in os.listdir(group_dir) if f.endswith(".txt")])
    return [os.path.join(group_dir, f) for f in files]


def load_instance_group(group):
    """Load every instance in the requested dataset group."""
    paths = _list_instances_in_group(group)
    bounds_index = _load_bounds_index()
    results = []
    for path in paths:
        inst_name = os.path.splitext(os.path.basename(path))[0]
        raw = parse_fjsp_txt(path)
        results.append(_inst_to_env(raw, inst_name, bounds_index))
    return results


# ── Makespan computation (core) ────────────────────────────────────


def _calc_makespan_core(seq_arr, ms_arr, env_data):
    """
    Validated FJSP schedule simulator with an equivalent array-only kernel.

    Args:
        seq_arr: np.ndarray of job IDs (dtype int32) length total_ops.
        ms_arr: np.ndarray (num_jobs, max_ops) of actual machine IDs.
        env_data: FJSP env dict with matrices.

    Returns:
        int makespan, or 999999999 if schedule is invalid.
    """
    num_jobs = env_data["num_jobs"]
    num_machines = env_data["num_machines"]
    processing_matrix = env_data["processing_matrix"]
    op_counts = env_data["op_counts"]

    total_ops = int(np.sum(op_counts))
    if len(seq_arr) != total_ops:
        return INVALID_SCORE
    if np.any(seq_arr < 0) or np.any(seq_arr >= num_jobs):
        return INVALID_SCORE
    if not np.all(np.bincount(seq_arr, minlength=num_jobs) == op_counts):
        return INVALID_SCORE
    if ms_arr.ndim != 2 or ms_arr.shape[0] < num_jobs or ms_arr.shape[1] < env_data["max_ops"]:
        return INVALID_SCORE

    return simulate_schedule(
        seq_arr, ms_arr, processing_matrix, op_counts, num_machines
    )


# ── Fallback MS construction ──────────────────────────────────────


def _build_fallback_ms(seq_arr, env_data):
    """
    Build a greedy machine selection for a bare sequence.

    For each (job, op), picks the eligible machine with the shortest
    processing time (SPT rule).  This guarantees a valid MS exists
    when the engine evaluates a sequence without a pre-built MS.
    """
    num_jobs = env_data["num_jobs"]
    op_counts = env_data["op_counts"]
    eligible_counts = env_data["eligible_counts"]
    times_matrix = env_data["times_matrix"]
    max_ops = env_data["max_ops"]

    ms = np.zeros((num_jobs, max_ops), dtype=np.int32)
    job_op_idx = np.zeros(num_jobs, dtype=np.int32)

    for pos in range(len(seq_arr)):
        job_id = int(seq_arr[pos])
        op_idx = int(job_op_idx[job_id])
        if op_idx < int(op_counts[job_id]):
            ec = int(eligible_counts[job_id, op_idx])
            # SPT: pick machine with minimum processing time
            best_k = 0
            best_t = 999999999
            for k in range(ec):
                t = int(times_matrix[job_id, op_idx, k])
                if t < best_t:
                    best_t = t
                    best_k = k
            ms[job_id, op_idx] = int(env_data["machines_matrix"][job_id, op_idx, best_k])
        job_op_idx[job_id] += 1

    return ms


# ── Public evaluation interface ────────────────────────────────────


def calc_makespan(sequence_or_state, env_data):
    """
    Compute FJSP makespan.

    Polymorphic calling convention:
      - Engine: calc_makespan(sequence_list, env_data)
        Uses a fallback greedy SPT machine selection when no MS is bound.
      - Slot functions: calc_makespan(state_obj, env_data)
        The state_obj has .sequence and .machine_selection attributes.

    Returns:
        float makespan (lower is better), or 999999999 for invalid.
    """
    if hasattr(sequence_or_state, "sequence"):
        seq = sequence_or_state.sequence
        metadata = getattr(sequence_or_state, "metadata", {}) or {}
        ms = metadata.get("machine_alloc")
        if ms is None:
            ms = getattr(sequence_or_state, "machine_selection", None)
    else:
        seq = sequence_or_state
        ms = get_active_machine_allocation()

    if seq is None:
        return INVALID_SCORE
    try:
        raw = np.asarray(seq).ravel()
        if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
            return INVALID_SCORE
        seq_arr = raw.astype(np.int32)
    except (TypeError, ValueError, OverflowError):
        return INVALID_SCORE

    total_ops = int(sum(env_data["op_counts"]))
    if len(seq_arr) != total_ops:
        return INVALID_SCORE
    if np.any(seq_arr < 0) or np.any(seq_arr >= env_data["num_jobs"]):
        return INVALID_SCORE
    if not np.all(np.bincount(seq_arr, minlength=env_data["num_jobs"]) == env_data["op_counts"]):
        return INVALID_SCORE

    if ms is None:
        ms = _build_fallback_ms(seq_arr, env_data)

    try:
        ms_arr = np.asarray(ms, dtype=np.int32)
    except (TypeError, ValueError, OverflowError):
        return INVALID_SCORE
    set_active_machine_allocation(ms_arr)
    return _calc_makespan_core(seq_arr, ms_arr, env_data)


# ── State initialisation ───────────────────────────────────────────


def initialize_state(env_data, state):
    """
    Ensure state has a feasible FJSP solution (OS + MS) if uninitialised.

    OS: random permutation of jobs (each job j repeated op_counts[j] times).
    MS: random eligible machine for each (job, operation).
    """
    if state.sequence is not None and len(state.sequence) > 0 and state.makespan < float("inf"):
        return  # Already valid

    num_jobs = env_data["num_jobs"]
    op_counts = env_data["op_counts"]
    eligible_counts = env_data["eligible_counts"]
    max_ops = env_data["max_ops"]

    # Build OS: random permutation
    seq_list = []
    for j in range(num_jobs):
        seq_list.extend([j] * int(op_counts[j]))
    random.shuffle(seq_list)
    seq = np.array(seq_list, dtype=np.int32)

    # Build MS: random eligible machine per operation
    ms = np.zeros((num_jobs, max_ops), dtype=np.int32)
    for j in range(num_jobs):
        for o in range(int(op_counts[j])):
            ec = int(eligible_counts[j, o])
            option = random.randint(0, ec - 1) if ec > 0 else 0
            ms[j, o] = int(env_data["machines_matrix"][j, o, option])

    state.sequence = seq
    state.machine_selection = ms
    state.metadata["machine_alloc"] = ms
    state.metadata["prev_alloc"] = ms.copy()
    set_active_machine_allocation(ms)
    state.makespan = _calc_makespan_core(seq, ms, env_data)


# ── Pipeline configuration ────────────────────────────────────────


def get_initial_pipeline():
    """Return the default FJSP operator pipeline."""
    return [
        "initialization|init_random",
        "sequence_perturbation|mut_sequence_swap",
        "machine_ls|ls_machine_greedy_balance",
        "acceptance|simulated_annealing",
    ]


def get_allowed_nodes(node_ablation_level):
    """Return the exact nested FJSP operator pool for a V-Level."""
    level_4 = {
        "acceptance|simulated_annealing",
        "general_ls|ls_adjacent",
        "initialization|init_random",
        "sequence_perturbation|mut_sequence_swap",
    }
    level_8 = level_4 | {
        "critical_path_ls|ls_n5_critical_block",
        "initialization|init_global_machine_load",
        "machine_ls|ls_machine_greedy_balance",
        "machine_perturbation|mut_machine_reassign",
    }
    level_15 = level_8 | {
        "coupled_perturbation|ruins_and_recreate",
        "critical_path_ls|ls_critical_machine_reassign",
        "critical_path_ls|ls_n7_critical_insert",
        "diversification|div_partial_reset",
        "diversification|div_random_walk",
        "machine_ls|ls_machine_idle_fill",
        "sequence_perturbation|mut_sequence_insert",
    }
    level_20 = level_15 | {
        "history_mining|history_elite_crossover",
        "initialization|init_alys_greedy",
        "intensification|tabu_search_lite",
        "intensification|vns_descent",
        "machine_perturbation|mut_bottleneck_roulette",
    }
    pools = {4: level_4, 8: level_8, 15: level_15, 20: level_20}
    try:
        level = int(node_ablation_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("FJSP node_ablation_level must be one of 4, 8, 15, or 20") from exc
    if level not in pools:
        raise ValueError(
            f"Unsupported FJSP node_ablation_level={node_ablation_level!r}; "
            "expected one of 4, 8, 15, or 20"
        )
    return set(pools[level])


# ── Domain prompt ─────────────────────────────────────────────────


def get_prompt(root_dir=None):
    """
    Read the FJSP domain knowledge prompt from disk.

    Args:
        root_dir: Project root directory. If None, auto-detected relative
                  to this file (problems/fjsp/ -> src/ -> root/).
    """
    if root_dir is None:
        root_dir = _get_project_root()
    prompt_path = os.path.join(root_dir, "prompts", "fjsp", "domain_knowledge.txt")
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
