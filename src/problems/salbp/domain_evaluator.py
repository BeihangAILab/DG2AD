"""
SALBP-1 domain evaluator — Simple Assembly Line Balancing Problem Type 1.

Exposes the DGA2D domain protocol:
  load_instance_group, get_reward_spec, describe_instance, calc_makespan,
  get_initial_pipeline, initialize_state, get_prompt

Problem: Given N tasks with processing times, precedence constraints, and a
fixed cycle time C, assign all tasks to the minimum number of sequential
workstations such that:
  1. Sum of task times at each station <= C
  2. If task u precedes task v, then station(u) <= station(v)

Encoding: Topological permutation of task IDs [0, 1, ..., N-1].
Decoder: Greedy First-Fit station assignment respecting precedence.

Objective: number of stations (minimisation problem).
"""

import os
import random
import numpy as np

from src.core.configuration import problem_data_dir

# ── Module-level config (set by Hydra before loading) ──
INVALID_SCORE = 99999.0
# The objective depends only on the permutation, not auxiliary state metadata.
SEQUENCE_ONLY_OBJECTIVE = True


def _parse_alb(filepath):
    """
    Parse a SALBP .alb instance file.

    Format:
      <number of tasks>      N
      <cycle time>           C
      <order strength>       OS (informational)
      <task times>           task_id  time  (1-indexed, N lines)
      <precedence relations> u,v  (1-indexed pairs until <end>)
      <end>
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    num_tasks = 0
    cycle_time = 0
    times = None
    edges = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("<"):
            i += 1
            continue

        if num_tasks == 0:
            num_tasks = int(line)
        elif cycle_time == 0:
            cycle_time = int(line)
        elif line.startswith("<order"):
            # Skip order strength line and its value
            i += 1
            # The next non-empty line might be the value
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines) and lines[i].strip().replace(".", "").isdigit():
                i += 1  # skip order strength value
            if line.startswith("<task"):
                # Read N task lines
                times = [0] * num_tasks
                for _ in range(num_tasks):
                    while i < len(lines) and not lines[i].strip():
                        i += 1
                    if i < len(lines):
                        parts = lines[i].strip().split()
                        task_id = int(parts[0]) - 1  # 0-indexed
                        t = int(parts[1])
                        times[task_id] = t
                        i += 1
            continue
        else:
            # Parse remaining sections
            pass

        i += 1

    # Second pass: parse the structured sections more robustly
    # Find task times section and precedence section by scanning
    section = None
    times = {}
    edges = []

    for line in lines:
        s = line.strip()
        if s.startswith("<task times>"):
            section = "times"
            continue
        elif s.startswith("<precedence relations>"):
            section = "prec"
            continue
        elif s.startswith("<end>"):
            section = None
            continue
        elif s.startswith("<"):
            section = None
            continue

        if not s:
            continue

        if section == "times":
            parts = s.split()
            if len(parts) >= 2:
                task_id = int(parts[0]) - 1
                times[task_id] = int(parts[1])
        elif section == "prec":
            parts = s.split(",")
            if len(parts) == 2:
                u = int(parts[0].strip()) - 1
                v = int(parts[1].strip()) - 1
                edges.append((u, v))

    # Also parse header info
    num_tasks = 0
    cycle_time = 0
    state = "num_tasks"
    for line in lines:
        s = line.strip()
        if not s or s.startswith("<"):
            continue
        if state == "num_tasks":
            num_tasks = int(s)
            state = "cycle_time"
        elif state == "cycle_time":
            cycle_time = int(s)
            state = "order_str"
        elif state == "order_str":
            state = "done"
            break

    # Build adjacency matrix
    adj_matrix = np.zeros((num_tasks, num_tasks), dtype=np.bool_)
    for u, v in edges:
        adj_matrix[u, v] = True

    # Convert times dict to array
    times_arr = np.zeros(num_tasks, dtype=np.int32)
    for tid, t in times.items():
        times_arr[tid] = t

    return {
        "num_tasks": num_tasks,
        "cycle_time": cycle_time,
        "times": times_arr,
        "adj_matrix": adj_matrix,
        "edges": np.array(edges, dtype=np.int32) if edges else np.zeros((0, 2), dtype=np.int32),
    }


# ── Instance discovery ───────────────────────────────────────────────


def _get_instances_dir():
    """Return the absolute path to the SALBP instances directory."""
    return str(problem_data_dir("salbp", os.path.join(os.path.dirname(__file__), "instances")))


# Group name mapping
_GROUP_DIRS = {
    "small": "small data set_n=20",
    "large": "large data set_n=100",
    "very_large": "very large data set_n=1000",
}


def _list_alb_files(group_str):
    """List all .alb files for a given group."""
    dir_name = _GROUP_DIRS.get(group_str, group_str)
    alb_dir = os.path.join(_get_instances_dir(), dir_name)
    if not os.path.isdir(alb_dir):
        # Try exact match
        for d in os.listdir(_get_instances_dir()):
            if group_str in d:
                alb_dir = os.path.join(_get_instances_dir(), d)
                break
    if not os.path.isdir(alb_dir):
        return []
    files = sorted(
        [
            (fname.replace(".alb", ""), os.path.join(alb_dir, fname))
            for fname in os.listdir(alb_dir)
            if fname.endswith(".alb")
        ]
    )
    return files


# ── Data loading (DGA2D domain protocol) ────────────────────────────


def load_instance_group(group):
    """Load every instance in the requested dataset group."""
    alb_files = _list_alb_files(group)
    if not alb_files:
        return []
    results = []
    for name, fpath in alb_files:
        results.append(_inst_to_env(name, fpath))
    return results


def _inst_to_env(name, filepath):
    """Convert a raw SALBP .alb instance to the standard env_data format."""
    parsed = _parse_alb(filepath)
    times = parsed["times"]
    cycle_time = parsed["cycle_time"]

    # Theoretical lower bound: ceil(sum(times) / cycle_time)
    total_time = float(np.sum(times))
    lb = int(np.ceil(total_time / cycle_time))

    # Use LB as upper_bound placeholder (no BKS available).
    # GAP = (makespan - LB) / LB * 100 shows how far above theoretical minimum.
    ub = float(lb)

    return {
        "name": name,
        "num_tasks": parsed["num_tasks"],
        "cycle_time": parsed["cycle_time"],
        "times": times,
        "adj_matrix": parsed["adj_matrix"],
        "edges": parsed["edges"],
        "upper_bound": ub,
        "lower_bound": ub,
        "total_time": total_time,
    }


# ── Solution evaluation ─────────────────────────────────────────────


try:
    from numba import njit
except ImportError:
    def njit(**kwargs):
        return lambda function: function


@njit(cache=True)
def _first_fit_stations(seq, times, adj, cycle_time):
    """Original topological validation and First-Fit decoder, compiled if available."""
    num_tasks = len(seq)
    # Validate topological order
    pos = np.zeros(num_tasks, dtype=np.int32)
    for i in range(num_tasks):
        pos[seq[i]] = i
    for u in range(num_tasks):
        for v in range(num_tasks):
            if adj[u, v] and pos[u] >= pos[v]:
                return INVALID_SCORE

    # Greedy First-Fit station assignment
    task_station = np.full(num_tasks, -1, dtype=np.int32)
    station_load = []  # list of accumulated times per station

    for task in seq:
        # Minimum station index = max station of predecessors
        min_st = 0
        for p in range(num_tasks):
            if adj[p, task] and task_station[p] > min_st:
                min_st = task_station[p]

        # Find earliest feasible station >= min_st
        assigned = False
        for s in range(min_st, len(station_load)):
            if station_load[s] + times[task] <= cycle_time:
                station_load[s] += times[task]
                task_station[task] = s
                assigned = True
                break

        if not assigned:
            station_load.append(times[task])
            task_station[task] = len(station_load) - 1

    return float(len(station_load))


def calc_makespan(sequence_or_state, env_data):
    """
    Compute SALBP-1 objective: number of workstations used.

    Decoder: Greedy First-Fit station assignment.
    For each task in sequence order:
      1. Determine the minimum feasible station:
         min_station = max(station[p] for all predecessors p)
      2. Find the first station s >= min_station where
         sum(task_times in s) + time[task] <= cycle_time.
      3. If no such station exists, open a new station.

    Validates that the sequence is a valid topological permutation.
    """
    num_tasks = env_data["num_tasks"]
    cycle_time = env_data["cycle_time"]
    times = env_data["times"]
    adj = env_data["adj_matrix"]

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

    if len(seq) != num_tasks:
        return INVALID_SCORE
    if seq.min() < 0 or seq.max() >= num_tasks:
        return INVALID_SCORE
    if len(set(seq)) != num_tasks:
        return INVALID_SCORE

    return _first_fit_stations(seq, times, adj, cycle_time)


# ── State initialisation ─────────────────────────────────────────────


def initialize_state(env_data, state):
    """
    Ensure state has a feasible topologically-sorted SALBP sequence.

    Uses Kahn's algorithm with random selection to generate diverse
    valid topological orderings.
    """
    if state.sequence is not None and len(state.sequence) > 0 and state.makespan < float("inf"):
        return

    n = env_data["num_tasks"]
    adj = env_data["adj_matrix"]

    best_seq = None
    best_mk = float("inf")

    for _ in range(20):
        in_degree = np.sum(adj, axis=0)
        eligible = [i for i in range(n) if in_degree[i] == 0]
        seq = []
        while eligible:
            idx = random.randrange(len(eligible))
            act = eligible.pop(idx)
            seq.append(act)
            for j in range(n):
                if adj[act, j]:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        eligible.append(j)
        arr = np.array(seq, dtype=np.int32)
        mk = calc_makespan(arr, env_data)
        if mk < best_mk:
            best_mk = mk
            best_seq = arr

    state.sequence = best_seq
    state.makespan = best_mk


# ── Pipeline configuration ───────────────────────────────────────────


def get_initial_pipeline():
    """Return the default SALBP-1 operator pipeline."""
    return [
        "initialization|init_random",
        "perturbation|mut_swap",
        "local_search|ls_swap",
        "acceptance|simulated_annealing",
    ]


# ── Domain prompt ────────────────────────────────────────────────────


def get_prompt(root_dir=None):
    """
    Read the SALBP domain knowledge prompt from disk.

    Args:
        root_dir: Project root directory. If None, auto-detected relative
                  to this file (problems/salbp/ -> src/ -> root/).
    """
    if root_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    prompt_path = os.path.join(root_dir, "prompts", "salbp", "domain_knowledge.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_reward_spec(env_data):
    from src.core.contracts import default_reward_spec

    return default_reward_spec(env_data)


def describe_instance(env_data):
    from src.core.contracts import default_describe_instance

    return default_describe_instance(env_data)
