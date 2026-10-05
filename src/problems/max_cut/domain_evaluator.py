"""
MAX_CUT domain evaluator.

Exposes the DGA2D domain protocol:
  load_instance_group, get_reward_spec, describe_instance, calc_makespan,
  get_initial_pipeline, initialize_state, get_prompt
"""

import os
import json
import random
import numpy as np
from numba import njit

from src.core.configuration import problem_data_dir

# ── Module-level config (set by Hydra before loading) ──
INVALID_SCORE = 999999999.0
GSET_LE2000_EXCLUDED = {"G50", "G55", "G60", "G65", "G70", "G72", "G81"}


def _load_bks_table():
    path = os.path.join(_get_instances_dir(), "gset_bks.json")
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)["best_known"]


def _get_bks(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    table = _load_bks_table()
    if name not in table:
        raise ValueError(
            f"No traceable G-set best-known value for {name}; "
            "add it to instances/gset_bks.json before reporting GAP"
        )
    return float(table[name])


@njit
def get_flip_delta(i, nodes, neighbors, weights, offsets):
    """Pure JIT-accelerated single-node flip delta calculation in CSR format."""
    side_i = nodes[i]
    delta = 0.0
    for idx in range(offsets[i], offsets[i + 1]):
        if nodes[neighbors[idx]] == side_i:
            delta += weights[idx]
        else:
            delta -= weights[idx]
    return delta


@njit
def calculate_maxcut_weight(nodes, neighbors, weights, offsets):
    """JIT-accelerated total cut weight evaluation in CSR format."""
    total_weight = 0.0
    num_nodes = len(nodes)
    for i in range(num_nodes):
        side_i = nodes[i]
        for idx in range(offsets[i], offsets[i + 1]):
            j = neighbors[idx]
            if nodes[j] != side_i:
                total_weight += weights[idx]
    return -(total_weight / 2.0)  # Negative for minimization compatibility


def load_maxcut_data(filepath):
    """
    Loads G-set format graphs and constructs adjacency lists in CSR format.
    """
    with open(filepath, "r") as f:
        line = f.readline().split()
        if not line:
            return None
        num_nodes, num_edges = int(line[0]), int(line[1])

        adj_temp = [[] for _ in range(num_nodes)]
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            w = float(parts[2]) if len(parts) > 2 else 1.0
            adj_temp[u - 1].append((v - 1, w))
            adj_temp[v - 1].append((u - 1, w))

    neighbors = []
    edge_weights = []
    offsets = [0]
    for i in range(num_nodes):
        for n, w in adj_temp[i]:
            neighbors.append(n)
            edge_weights.append(w)
        offsets.append(len(neighbors))

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "neighbors": np.ascontiguousarray(neighbors, dtype=np.int32),
        "edge_weights": np.ascontiguousarray(edge_weights, dtype=np.float32),
        "offsets": np.ascontiguousarray(offsets, dtype=np.int32),
    }


def _get_instances_dir():
    return str(problem_data_dir("max_cut", os.path.join(os.path.dirname(__file__), "instances")))


def _resolve_instance_path(filename):
    """Dynamically resolve instance path relative to this file."""
    return os.path.join(_get_instances_dir(), filename)


# ── Data loading ──────────────────────────────────────────────────


def load_instance_group(group):
    """Load every instance in the requested dataset group."""
    if group not in {"G-set", "G-set_le2000"}:
        raise KeyError(f"Unknown Max-Cut benchmark group: {group}")
    inst_dir = _get_instances_dir()

    # Sort files numerically if they follow the G*.txt convention
    def get_num(fname):
        try:
            return int(fname[1:-4])
        except Exception:
            return 999999

    files = sorted(
        [f for f in os.listdir(inst_dir) if f.startswith("G") and f.endswith(".txt")], key=get_num
    )

    if group == "G-set_le2000":
        files = [
            filename
            for filename in files
            if os.path.splitext(filename)[0] not in GSET_LE2000_EXCLUDED
        ]

    res = []
    for f in files:
        path = os.path.join(inst_dir, f)
        env = load_maxcut_data(path)
        if env:
            bks = _get_bks(f)
            env["opt_cost"] = -bks
            env["lower_bound"] = -bks
            env["upper_bound"] = -bks
            env["best_known_cut"] = bks
            env["name"] = f
            res.append(env)
    return res


# ── Solution evaluation (makespan = -total cut weight) ─────────────


def calc_makespan(sequence_or_state, env_data):
    """
    Compute total cut weight (returned as negative value for minimization).
    """
    sequence = (
        sequence_or_state.sequence if hasattr(sequence_or_state, "sequence") else sequence_or_state
    )
    if sequence is None:
        return INVALID_SCORE
    try:
        raw = np.asarray(sequence).ravel()
        if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
            return INVALID_SCORE
        nodes = np.ascontiguousarray(raw.astype(np.int32))
    except (TypeError, ValueError, OverflowError):
        return INVALID_SCORE
    if len(nodes) != env_data["num_nodes"]:
        return INVALID_SCORE
    if np.any((nodes != 0) & (nodes != 1)):
        return INVALID_SCORE

    return float(
        calculate_maxcut_weight(
            nodes, env_data["neighbors"], env_data["edge_weights"], env_data["offsets"]
        )
    )


# ── State initialisation ──────────────────────────────────────────


def initialize_state(env_data, state):
    """
    Initialize a random partition nodes.
    """
    if state.sequence and state.makespan < float("inf"):
        return

    num_nodes = env_data["num_nodes"]
    seq = [random.choice([0, 1]) for _ in range(num_nodes)]
    state.sequence = seq
    state.makespan = calc_makespan(seq, env_data)


# ── Pipeline configuration ────────────────────────────────────────


def get_initial_pipeline():
    """Return the default Max-Cut operator pipeline."""
    return [
        "initialization|init_random",
        "local_search|ls_maxcut",
        "perturbation|mutation_maxcut",
        "intensification|accept_maxcut",
    ]


def get_allowed_nodes(node_ablation_level):
    return {
        "initialization|init_random",
        "local_search|ls_maxcut",
        "perturbation|mutation_maxcut",
        "intensification|accept_maxcut",
    }


# ── Domain prompt ─────────────────────────────────────────────────


def get_prompt(root_dir=None):
    """
    Read the Max-Cut domain knowledge prompt from disk.
    """
    if root_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    prompt_path = os.path.join(root_dir, "prompts", "max_cut", "domain_knowledge.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_reward_spec(env_data):
    from src.core.contracts import default_reward_spec

    return default_reward_spec(env_data)


def describe_instance(env_data):
    from src.core.contracts import default_describe_instance

    return default_describe_instance(env_data)
