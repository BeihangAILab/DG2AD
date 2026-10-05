"""
RCPSP domain evaluator — Resource-Constrained Project Scheduling Problem.

Exposes the DGA2D domain protocol:
  load_instance_group, get_reward_spec, describe_instance, calc_makespan,
  get_initial_pipeline, initialize_state, get_prompt

Encoding: topological ordering of N activities [0, 1, ..., N-1].
  - Activity 0 is the dummy source (duration 0).
  - Activity N-1 is the dummy sink (duration 0).
  - Sequence must respect all precedence constraints.

Decoder: Serial Schedule Generation Scheme (SGS).
  For each activity in sequence order, schedule it at the earliest time
  where all predecessors are finished AND sufficient resource capacity is
  available for the activity's entire duration.

Objective: makespan (finish time of the dummy sink).
  THIS IS A MINIMISATION PROBLEM.
"""

import json
import os
import random
import numpy as np

from src.core.configuration import problem_data_dir

INVALID_SCORE = 99999999.0

# Optional numba acceleration
try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        """No-op decorator when numba is unavailable."""

        def wrapper(fn):
            return fn

        return wrapper


# ── Module-level config (set by Hydra before loading) ──


def _parse_psplib_sm(filepath):
    """
    Parse a PSPLIB standard .sm file for RCPSP.

    Returns a dict with keys: num_activities, num_resources, durations,
    requests, capacities, successors, adj_matrix, edges.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    num_activities = 0
    num_resources = 0
    successors = None
    durations = None
    requests = None
    capacities = None

    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()

        if "jobs (incl. supersource/sink" in line:
            parts = line.split()
            num_activities = int(parts[-1])

        if "- renewable" in line:
            parts = line.split()
            num_resources = int(parts[3])

        if "PRECEDENCE RELATIONS:" in line:
            idx += 2
            successors = [[] for _ in range(num_activities)]
            for _ in range(num_activities):
                parts = list(map(int, lines[idx].split()))
                act_id = parts[0] - 1
                num_succ = parts[2]
                for s in parts[3 : 3 + num_succ]:
                    successors[act_id].append(s - 1)
                idx += 1

        if "REQUESTS/DURATIONS:" in line:
            idx += 3
            durations = np.zeros(num_activities, dtype=np.int32)
            requests = np.zeros((num_activities, num_resources), dtype=np.int32)
            for _ in range(num_activities):
                parts = list(map(int, lines[idx].split()))
                act_id = parts[0] - 1
                durations[act_id] = parts[2]
                requests[act_id] = np.array(parts[3 : 3 + num_resources], dtype=np.int32)
                idx += 1

        if "RESOURCEAVAILABILITIES:" in line:
            idx += 2
            capacities = np.array(list(map(int, lines[idx].split())), dtype=np.int32)
            break

        idx += 1

    # Build adjacency matrix and edge list
    adj_matrix = np.zeros((num_activities, num_activities), dtype=np.bool_)
    edges_list = []
    for i, succs in enumerate(successors):
        for s in succs:
            adj_matrix[i, s] = True
            edges_list.append((i, s))
    edges = np.array(edges_list, dtype=np.int32) if edges_list else np.zeros((0, 2), dtype=np.int32)

    return {
        "num_activities": num_activities,
        "num_resources": num_resources,
        "durations": durations,
        "requests": requests,
        "capacities": capacities,
        "adj_matrix": adj_matrix,
        "edges": edges,
    }


# ── Topological order validator ──────────────────────────────────────


@njit
def _check_topological_order(sequence, edges, num_activities):
    """
    Verify that sequence respects all precedence constraints (O(|E|)).

    For each edge (u -> v), u must appear before v in the sequence.
    """
    pos = np.zeros(num_activities, dtype=np.int32)
    for i in range(num_activities):
        pos[sequence[i]] = i

    for k in range(edges.shape[0]):
        u = edges[k, 0]
        v = edges[k, 1]
        if pos[u] >= pos[v]:
            return False
    return True


# ── Serial SGS decoder ───────────────────────────────────────────────


@njit
def _serial_sgs(
    sequence, num_activities, num_resources, durations, requests, capacities, adj_matrix
):
    """
    Serial Schedule Generation Scheme for RCPSP.

    Processes activities in the given sequence order. Each activity is
    scheduled at the earliest time t where:
      1. All predecessors have finished (precedence constraint).
      2. For each time unit [t, t + dur), resource usage fits within
         resource capacities (resource constraint).

    Returns the makespan (finish time of the last activity).
    """
    start_times = np.zeros(num_activities, dtype=np.int32)
    finish_times = np.zeros(num_activities, dtype=np.int32)
    max_time = np.sum(durations) + 1
    resource_usage = np.zeros((max_time, num_resources), dtype=np.int32)

    for i in range(num_activities):
        act = sequence[i]

        # Earliest start based on predecessors
        early_start = 0
        for p in range(num_activities):
            if adj_matrix[p, act]:
                if finish_times[p] > early_start:
                    early_start = finish_times[p]

        t = early_start
        dur = durations[act]
        req = requests[act]

        # Find earliest feasible time satisfying resource constraints
        while t < max_time:
            fit = True
            for time_offset in range(dur):
                curr_t = t + time_offset
                if curr_t >= max_time:
                    fit = False
                    break
                for r in range(num_resources):
                    if resource_usage[curr_t, r] + req[r] > capacities[r]:
                        fit = False
                        break
                if not fit:
                    break

            if fit:
                start_times[act] = t
                finish_times[act] = t + dur
                for time_offset in range(dur):
                    for r in range(num_resources):
                        resource_usage[t + time_offset, r] += req[r]
                break
            t += 1

    return np.max(finish_times)


# ── Instance discovery ───────────────────────────────────────────────


def _get_instances_dir():
    """Return the absolute path to the RCPSP instances directory."""
    return str(problem_data_dir("rcpsp", os.path.join(os.path.dirname(__file__), "instances")))


def _list_sm_files(group_str):
    """List all .sm files for a given group (e.g. 'j30', 'j60', 'j90', 'j120')."""
    sm_dir = os.path.join(_get_instances_dir(), group_str + ".sm")
    if not os.path.isdir(sm_dir):
        return []
    files = sorted(
        [
            (fname.replace(".sm", ""), os.path.join(sm_dir, fname))
            for fname in os.listdir(sm_dir)
            if fname.endswith(".sm")
        ]
    )
    return files


def _load_bks_from_merged(group_str):
    """Load upper_bound from the merged JSON for a given group."""
    merged_path = os.path.join(_get_instances_dir(), f"{group_str}_sm_merged.json")
    if not os.path.exists(merged_path):
        return {}
    with open(merged_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bks = {}
    for sol in data.get("solutions", []):
        param = sol["parameter"]
        inst = sol["instance"]
        name = f"j{group_str[1:]}{param}_{inst}"
        ub = sol.get("optimal_makespan") or sol.get("upper_bound")
        if ub is not None:
            bks[name] = float(ub)
    return bks


def _load_bks_from_json():
    """Load BKS values from rcpsp_bks.json."""
    bks_path = os.path.join(_get_instances_dir(), "rcpsp_bks.json")
    if os.path.exists(bks_path):
        with open(bks_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _lookup_upper_bound(name, group_str):
    """Look up the best-known makespan for a given instance."""
    # Try merged JSON first
    merged_bks = _load_bks_from_merged(group_str)
    if name in merged_bks:
        return merged_bks[name]
    # Try rcpsp_bks.json
    bks_json = _load_bks_from_json()
    if name in bks_json:
        return float(bks_json[name])
    return None


# ── Data loading (DGA2D domain protocol) ────────────────────────────


def load_instance_group(group):
    """Load every instance in the requested dataset group."""
    sm_files = _list_sm_files(group)
    if not sm_files:
        return []
    results = []
    for name, fpath in sm_files:
        results.append(_inst_to_env(name, fpath, group))
    return results


def _inst_to_env(name, filepath, group_str):
    """Convert a raw RCPSP .sm instance to the standard env_data format."""
    parsed = _parse_psplib_sm(filepath)
    ub = _lookup_upper_bound(name, group_str)
    if ub is None:
        ub = float(np.sum(parsed["durations"]))

    return {
        "name": name,
        "group": group_str,
        "num_activities": parsed["num_activities"],
        "num_resources": parsed["num_resources"],
        "durations": parsed["durations"],
        "requests": parsed["requests"],
        "capacities": parsed["capacities"],
        "adj_matrix": parsed["adj_matrix"],
        "edges": parsed["edges"],
        "upper_bound": ub,
        "lower_bound": ub,
    }


# ── Solution evaluation ─────────────────────────────────────────────


def calc_makespan(sequence_or_state, env_data):
    """
    Compute RCPSP makespan via Serial SGS.

    Validates topological order: if any precedence constraint is violated,
    returns a penalty value (99999999.0).

    Expects a permutation of [0, 1, ..., num_activities-1] with activity 0
    (dummy source) first and activity N-1 (dummy sink) last — but the
    decoder handles any topological order.
    """
    num_activities = env_data["num_activities"]
    num_resources = env_data["num_resources"]
    edges = env_data["edges"]

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

    if len(seq) != num_activities:
        return INVALID_SCORE
    if seq.min() < 0 or seq.max() >= num_activities:
        return INVALID_SCORE
    if len(set(seq)) != num_activities:
        return INVALID_SCORE

    # Validate precedence constraints
    if not _check_topological_order(seq, edges, num_activities):
        return INVALID_SCORE

    return float(
        _serial_sgs(
            seq,
            num_activities,
            num_resources,
            env_data["durations"],
            env_data["requests"],
            env_data["capacities"],
            env_data["adj_matrix"],
        )
    )


# ── State initialisation ─────────────────────────────────────────────


def initialize_state(env_data, state):
    """
    Ensure state has a feasible (topologically-ordered) RCPSP sequence.

    Generates a random topological ordering using Kahn's algorithm.
    """
    if state.sequence is not None and len(state.sequence) > 0 and state.makespan < float("inf"):
        return

    n = env_data["num_activities"]
    adj = env_data["adj_matrix"]

    # Compute in-degrees
    in_degree = np.sum(adj, axis=0)
    eligible = [i for i in range(n) if in_degree[i] == 0]

    seq = []
    while eligible:
        # Pick a random eligible activity
        idx = random.randrange(len(eligible))
        act = eligible.pop(idx)
        seq.append(act)
        # Update in-degrees for successors
        for j in range(n):
            if adj[act, j]:
                in_degree[j] -= 1
                if in_degree[j] == 0:
                    eligible.append(j)

    state.sequence = np.array(seq, dtype=np.int32)
    state.makespan = calc_makespan(state, env_data)


# ── Pipeline configuration ───────────────────────────────────────────


def get_initial_pipeline():
    """Return the default RCPSP operator pipeline."""
    return [
        "initialization|init_random",
        "perturbation|mut_swap",
        "local_search|ls_swap",
        "acceptance|simulated_annealing",
    ]


# ── Domain prompt ────────────────────────────────────────────────────


def get_prompt(root_dir=None):
    """
    Read the RCPSP domain knowledge prompt from disk.

    Args:
        root_dir: Project root directory. If None, auto-detected relative
                  to this file (problems/rcpsp/ -> src/ -> root/).
    """
    if root_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    prompt_path = os.path.join(root_dir, "prompts", "rcpsp", "domain_knowledge.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_reward_spec(env_data):
    from src.core.contracts import default_reward_spec

    return default_reward_spec(env_data)


def describe_instance(env_data):
    from src.core.contracts import default_describe_instance

    return default_describe_instance(env_data)
