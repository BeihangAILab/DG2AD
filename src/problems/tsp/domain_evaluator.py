"""
TSP domain evaluator — Travelling Salesman Problem.

Exposes the DGA2D domain protocol:
  load_instance_group, get_reward_spec, describe_instance, calc_makespan,
  get_initial_pipeline, initialize_state, get_prompt

Encoding: permutation (tour) of city indices [0, 1, ..., N-1].
  The tour visits cities in order and returns to the start.

Distance: EUC_2D (Euclidean 2D, rounded to nearest integer per TSPLIB).
  dist[i,j] = int(sqrt((x_i-x_j)^2 + (y_i-y_j)^2) + 0.5)

Objective: total tour length (minimisation problem).
"""

import json
import math
import os
import random
import numpy as np

from src.core.configuration import problem_data_dir

# ── Module-level config (set by Hydra before loading) ──
INVALID_SCORE = 999999999.0


def _parse_tsplib(filepath):
    """
    Parse a TSPLIB standard symmetric TSP .tsp file.

    Returns:
        coords: np.ndarray of shape (N, 2) — [x, y] per city
        name: instance name string
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    coords = []
    name = ""
    in_coord_section = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("NAME"):
            parts = line.split(":")
            name = parts[1].strip() if len(parts) > 1 else line.split()[-1].strip()
        elif line.startswith("DIMENSION"):
            parts = line.split(":")
            if len(parts) > 1:
                int(parts[1].strip())
            else:
                int(line.split()[-1].strip())
        elif line.startswith("NODE_COORD_SECTION"):
            in_coord_section = True
            continue
        elif line == "EOF":
            break

        if in_coord_section:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    coords.append([x, y])
                except ValueError:
                    pass

    coords = np.array(coords, dtype=np.float64)
    return coords, name


# ── Distance computation ──────────────────────────────────────────────


def _compute_distance_matrix(coords):
    """
    Compute the EUC_2D distance matrix (TSPLIB standard).

    dist[i,j] = int(sqrt((dx)^2 + (dy)^2) + 0.5)
    """
    n = len(coords)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i, j] = 0.0
            else:
                dx = coords[i, 0] - coords[j, 0]
                dy = coords[i, 1] - coords[j, 1]
                dist[i, j] = float(int(math.sqrt(dx * dx + dy * dy) + 0.5))
    return dist


# ── Instance discovery ───────────────────────────────────────────────


def _get_instances_dir():
    """Return the absolute path to the TSP instances directory."""
    return str(problem_data_dir("tsp", os.path.join(os.path.dirname(__file__), "instances")))


def _list_tsp_files(group_str):
    """List all .tsp files for a given group (subdirectory name)."""
    tsp_dir = os.path.join(_get_instances_dir(), group_str)
    if not os.path.isdir(tsp_dir):
        return []
    files = sorted(
        [
            (fname.replace(".tsp", ""), os.path.join(tsp_dir, fname))
            for fname in os.listdir(tsp_dir)
            if fname.endswith(".tsp")
        ]
    )
    return files


def _load_bks():
    """Load best-known solution values from tsp_bks.json."""
    bks_path = os.path.join(_get_instances_dir(), "tsp_bks.json")
    if os.path.exists(bks_path):
        with open(bks_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _lookup_upper_bound(name, coords):
    """Look up the BKS tour length for a given instance name."""
    bks = _load_bks()
    # Try exact name match, then lowercase match
    if name in bks:
        return float(bks[name])
    if name.lower() in bks:
        return float(bks[name.lower()])
    return None


# ── Data loading (DGA2D domain protocol) ────────────────────────────


def load_instance_group(group):
    """Load every instance in the requested dataset group."""
    tsp_files = _list_tsp_files(group)
    if not tsp_files:
        return []
    results = []
    for name, fpath in tsp_files:
        results.append(_inst_to_env(name, fpath))
    return results


def _inst_to_env(name, filepath):
    """Convert a raw TSP .tsp instance to the standard env_data format."""
    coords, parsed_name = _parse_tsplib(filepath)
    if parsed_name:
        name = parsed_name
    dist = _compute_distance_matrix(coords)
    ub = _lookup_upper_bound(name, coords)

    n = len(coords)
    if ub is None:
        ub = float(n * 1000)  # fallback for unknown instances

    return {
        "name": name,
        "num_nodes": n,
        "coords": coords,
        "distance_matrix": dist,
        "upper_bound": ub,
        "lower_bound": ub,
    }


# ── Solution evaluation ─────────────────────────────────────────────


def calc_makespan(sequence_or_state, env_data):
    """
    Compute total TSP tour length.

    The tour visits cities in the order given by the sequence,
    then returns to the first city.

    Validates that the sequence contains exactly one of each city index.
    """
    dist = env_data["distance_matrix"]
    n = env_data["num_nodes"]

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

    if len(seq) != n:
        return INVALID_SCORE
    if seq.min() < 0 or seq.max() >= n:
        return INVALID_SCORE
    if len(set(seq)) != n:
        return INVALID_SCORE

    total = 0.0
    for i in range(n):
        u = int(seq[i])
        v = int(seq[(i + 1) % n])
        total += dist[u, v]

    return float(total)


# ── State initialisation ─────────────────────────────────────────────


def initialize_state(env_data, state):
    """
    Ensure state has a valid TSP tour.

    Uses Nearest Neighbour from multiple random starting cities
    and returns the best tour found.
    """
    if state.sequence is not None and len(state.sequence) > 0 and state.makespan < float("inf"):
        return

    n = env_data["num_nodes"]
    dist = env_data["distance_matrix"]

    num_starts = min(5, n)
    start_cities = random.sample(range(n), num_starts)

    best_seq = None
    best_cost = float("inf")

    for start in start_cities:
        visited = [False] * n
        seq = [start]
        visited[start] = True
        current = start

        for _ in range(n - 1):
            row = dist[current]
            # Find nearest unvisited city
            best_next = -1
            best_d = float("inf")
            for j in range(n):
                if not visited[j] and row[j] < best_d:
                    best_d = row[j]
                    best_next = j
            if best_next == -1:
                break
            seq.append(best_next)
            visited[best_next] = True
            current = best_next

        cost = calc_makespan(np.array(seq, dtype=np.int32), env_data)
        if cost < best_cost:
            best_cost = cost
            best_seq = seq

    if best_seq is None:
        best_seq = list(range(n))
        best_cost = calc_makespan(np.array(best_seq, dtype=np.int32), env_data)

    state.sequence = np.array(best_seq, dtype=np.int32)
    state.makespan = best_cost


# ── Pipeline configuration ───────────────────────────────────────────


def get_initial_pipeline():
    """Return the default TSP operator pipeline."""
    return [
        "initialization|init_nearest_neighbor",
        "perturbation|mut_swap",
        "local_search|ls_2opt",
        "acceptance|simulated_annealing",
    ]


# ── Domain prompt ────────────────────────────────────────────────────


def get_prompt(root_dir=None):
    """
    Read the TSP domain knowledge prompt from disk.

    Args:
        root_dir: Project root directory. If None, auto-detected relative
                  to this file (problems/tsp/ -> src/ -> root/).
    """
    if root_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    prompt_path = os.path.join(root_dir, "prompts", "tsp", "domain_knowledge.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_reward_spec(env_data):
    from src.core.contracts import default_reward_spec

    return default_reward_spec(env_data)


def describe_instance(env_data):
    from src.core.contracts import default_describe_instance

    return default_describe_instance(env_data)
