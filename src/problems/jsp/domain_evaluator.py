"""
JSP domain evaluator — Job-shop Scheduling Problem.

Exposes the DGA2D domain protocol:
  load_instance_group, get_reward_spec, describe_instance, calc_makespan,
  get_initial_pipeline, initialize_state, get_prompt

Prompt text is read from prompts/jsp/domain_knowledge.txt.

JSP encoding:
  - state.sequence: flat list / array of length N * M.
    Each job j appears exactly M times; the k-th occurrence = job j's k-th
    operation, whose machine and time are read from machines_matrix[j][k]
    and times_matrix[j][k].
  - No machine_selection needed (routing is fixed per job).
"""

import os
import json
import random
import numpy as np

from src.core.configuration import problem_data_dir

# ── Module-level config (set by Hydra before loading) ──
INVALID_SCORE = 999999999.0


def _get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _get_instances_path():
    private_fallback = os.path.join(_get_project_root(), "src", "problems", "jsp", "instances")
    return str(problem_data_dir("jsp", private_fallback) / "jsp_instances.json")


# ── Instance loading ──────────────────────────────────────────────


def _load_all_instances():
    path = _get_instances_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSP instances file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return [e for e in entries if "machines_matrix" in e]


def _inst_to_env(inst):
    nj, nm = inst["n_jobs"], inst["n_machines"]
    return {
        "num_jobs": nj,
        "num_machines": nm,
        "total_ops": nj * nm,
        "machines_matrix": np.array(inst["machines_matrix"], dtype=np.int32),
        "times_matrix": np.array(inst["times_matrix"], dtype=np.int32),
        "lower_bound": float(inst.get("lower_bound", 1)),
        "upper_bound": float(inst.get("upper_bound", 1)),
        "name": inst["name"],
        "group": inst.get("group", ""),
    }


def load_instance_group(group):
    """Load every instance in the requested dataset group."""
    all_insts = _load_all_instances()
    group_insts = [e for e in all_insts if e.get("group") == group]
    return [_inst_to_env(inst) for inst in group_insts]


# ── Makespan computation (core) ───────────────────────────────────


def _calc_makespan_core(seq_arr, env_data):
    """
    Pure Python JSP schedule simulator.

    For each operation in seq_arr:
      1. Look up its job ID.
      2. Count how many times this job has appeared → operation index k.
      3. machine = machines_matrix[job][k], time = times_matrix[job][k].
      4. start = max(job_ready[job], machine_available[machine]).
      5. Schedule and update availability.
    """
    nj = env_data["num_jobs"]
    nm = env_data["num_machines"]
    machines = env_data["machines_matrix"]
    times = env_data["times_matrix"]

    if len(seq_arr) != nj * nm:
        return INVALID_SCORE
    if np.any(seq_arr < 0) or np.any(seq_arr >= nj):
        return INVALID_SCORE
    if not np.all(np.bincount(seq_arr, minlength=nj) == nm):
        return INVALID_SCORE

    job_op_idx = np.zeros(nj, dtype=np.int32)
    machine_avail = np.zeros(nm, dtype=np.float64)
    job_ready = np.zeros(nj, dtype=np.float64)

    for pos in range(len(seq_arr)):
        job_id = int(seq_arr[pos])
        op_idx = int(job_op_idx[job_id])

        if op_idx >= nm:
            return INVALID_SCORE

        m = int(machines[job_id, op_idx])
        t = float(times[job_id, op_idx])

        start = max(job_ready[job_id], machine_avail[m])
        end = start + t

        machine_avail[m] = end
        job_ready[job_id] = end
        job_op_idx[job_id] += 1

    return float(max(machine_avail))


# ── Public evaluation interface ───────────────────────────────────


def calc_makespan(sequence_or_state, env_data):
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
    """Ensure state has a valid JSP sequence (random permutation)."""
    if state.sequence is not None and len(state.sequence) > 0 and state.makespan < float("inf"):
        return

    nj, nm = env_data["num_jobs"], env_data["num_machines"]
    base = [j for j in range(nj) for _ in range(nm)]
    random.shuffle(base)
    seq = np.array(base, dtype=np.int32)
    state.sequence = seq
    state.makespan = _calc_makespan_core(seq, env_data)


# ── Pipeline configuration ────────────────────────────────────────


def get_initial_pipeline():
    return [
        "initialization|init_greedy",
        "perturbation|ruins_recreate",
        "local_search|ls_n5_critical",
        "acceptance|simulated_annealing",
    ]


# ── Domain prompt ─────────────────────────────────────────────────


def get_prompt(root_dir=None):
    if root_dir is None:
        root_dir = _get_project_root()
    prompt_path = os.path.join(root_dir, "prompts", "jsp", "domain_knowledge.txt")
    if not os.path.exists(prompt_path):
        return ""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


# ── Critical path extraction (used by ls_n5_critical slot) ────────


def extract_critical_path(seq, env_data):
    """
    Trace the critical path of a JSP schedule.

    Returns a list of (job, op) tuples on the critical path, ordered from
    the last operation backwards.  Used by the N5 neighbourhood operator.
    """
    nj = env_data["num_jobs"]
    nm = env_data["num_machines"]
    machines = env_data["machines_matrix"]
    times = env_data["times_matrix"]

    n = len(seq)
    job_op = np.zeros(nj, dtype=np.int32)
    job_end = np.zeros(nj, dtype=np.float64)
    mach_end = np.zeros(nm, dtype=np.float64)

    # Forward schedule
    sched_m = np.zeros(n, dtype=np.int32)
    sched_s = np.zeros(n, dtype=np.float64)
    sched_e = np.zeros(n, dtype=np.float64)
    pred_j = np.zeros(n, dtype=np.int8)  # blocked by same-job predecessor
    pred_m = np.zeros(n, dtype=np.int8)  # blocked by same-machine predecessor
    job_to_idx = np.full((nj, nm), -1, dtype=np.int32)

    for pos in range(n):
        job = int(seq[pos])
        op = int(job_op[job])

        if op >= nm:
            return []

        m = int(machines[job, op])
        t = float(times[job, op])

        je = job_end[job]
        me = mach_end[m]
        start = je if je > me else me
        end = start + t

        sched_m[pos] = m
        sched_s[pos] = start
        sched_e[pos] = end
        pred_j[pos] = abs(je - start) < 1e-9
        pred_m[pos] = abs(me - start) < 1e-9

        job_end[job] = end
        mach_end[m] = end
        job_to_idx[job, op] = pos
        job_op[job] += 1

    makespan = float(max(mach_end))

    # Find the job+op that finishes last
    crit_job = -1
    crit_op = -1
    for j in range(nj):
        if abs(job_end[j] - makespan) < 1e-9:
            crit_job = j
            crit_op = int(job_op[j]) - 1
            break

    if crit_job == -1:
        return []

    # Walk backwards
    visited = np.zeros(n, dtype=np.int8)
    path = []

    while True:
        idx = int(job_to_idx[crit_job, crit_op])
        if idx == -1 or visited[idx]:
            break
        visited[idx] = 1
        path.append((crit_job, crit_op))

        pj = bool(pred_j[idx])
        pm = bool(pred_m[idx])

        if crit_op > 0 and pj:
            crit_op -= 1
        else:
            found = False
            cur_m = int(sched_m[idx])
            cur_s = float(sched_s[idx])
            if pm:
                for idx2 in range(n):
                    if sched_m[idx2] == cur_m and abs(sched_e[idx2] - cur_s) < 1e-9:
                        crit_job2 = int(seq[idx2])
                        for temp_o in range(nm):
                            if job_to_idx[crit_job2, temp_o] == idx2:
                                crit_job = crit_job2
                                crit_op = temp_o
                                found = True
                                break
                        if found:
                            break
            if not found:
                break

    return path


def get_reward_spec(env_data):
    from src.core.contracts import default_reward_spec

    return default_reward_spec(env_data)


def describe_instance(env_data):
    from src.core.contracts import default_describe_instance

    return default_describe_instance(env_data)
