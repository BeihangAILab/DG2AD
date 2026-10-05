"""
CVRP domain evaluator — Capacitated Vehicle Routing Problem.

Exposes the DGA2D domain protocol:
  load_instance_group, get_reward_spec, describe_instance, calc_makespan,
  get_initial_pipeline, initialize_state, get_prompt

Prompt text is read from prompts/cvrp/domain_knowledge.txt.
"""

import os
import random
import numpy as np

from src.utils.cvrp_utils import load_instance_set

# ── Module-level config (set by Hydra before loading) ──
INVALID_SCORE = 999999999.0


def load_instance_group(group):
    """Load every instance in the requested dataset group."""
    instances = load_instance_set(group)
    return [_inst_to_env(inst) for inst in instances]


def _inst_to_env(inst):
    """Convert a raw CVRP instance dict to the standard env_data format."""
    n = inst["n_nodes"]
    depot = inst["depot_idx"]
    opt = inst.get("opt_cost", 1) or 1
    return {
        "dist_matrix": inst["dist_matrix"],
        "demands": inst["demands"],
        "capacity": inst["capacity"],
        "num_nodes": n,
        "num_customers": n - 1,
        "depot_idx": depot,
        "opt_cost": opt,
        "lower_bound": float(opt),
        "upper_bound": float(opt),
        "name": inst["name"],
    }


# ── Solution evaluation (makespan = total route distance) ─────────


def _parse_routes_from_sequence(sequence, env_data):
    """
    Parse a flat sequence into a list of routes.

    Sequence format: [c1, c2, ..., 0, c3, c4, ..., 0, ...]
    where 0 is the depot / route separator.
    Customer IDs are 1-indexed (1 to n-1).
    """
    routes = []
    current_route = []
    for x in sequence:
        if x == 0:
            if current_route:
                routes.append(current_route)
                current_route = []
        else:
            current_route.append(x)
    if current_route:
        routes.append(current_route)
    return routes


def calc_makespan(sequence_or_state, env_data):
    """
    Compute total route distance from a sequence representation.

    Performs anti-cheating validation:
      1. All customers visited exactly once.
      2. No capacity violations.
      3. No out-of-range node IDs.
    """
    sequence = (
        sequence_or_state.sequence if hasattr(sequence_or_state, "sequence") else sequence_or_state
    )
    n = env_data["num_nodes"]
    num_customers = env_data["num_customers"]
    capacity = env_data["capacity"]
    dist = env_data["dist_matrix"]
    demands = env_data["demands"]
    depot = env_data["depot_idx"]

    if not isinstance(sequence, (list, np.ndarray)):
        return INVALID_SCORE

    routes = _parse_routes_from_sequence(sequence, env_data)

    customer_set = set()
    for route in routes:
        if not route:
            return INVALID_SCORE
        for c in route:
            if c <= 0 or c >= n:
                return INVALID_SCORE
            if c in customer_set:
                return INVALID_SCORE
            customer_set.add(c)

    if len(customer_set) != num_customers:
        return INVALID_SCORE

    for route in routes:
        load = sum(demands[c] for c in route)
        if load > capacity + 1e-6:
            return INVALID_SCORE

    total = 0.0
    for route in routes:
        prev = depot
        for c in route:
            total += dist[prev, c]
            prev = c
        total += dist[prev, depot]

    return float(total)


# ── State initialisation ──────────────────────────────────────────


def initialize_state(env_data, state):
    """
    Ensure state has a feasible CVRP solution if uninitialised.

    Uses a greedy bin-packing-style construction: randomly shuffle customers,
    then fill routes sequentially respecting capacity. Depot (0) inserted
    as a route separator.
    """
    if state.sequence and state.makespan < float("inf"):
        return  # Already valid

    demands = env_data["demands"]
    capacity = env_data["capacity"]
    num_customers = env_data["num_customers"]

    customers = list(range(1, num_customers + 1))
    random.shuffle(customers)

    seq = []
    curr_load = 0.0
    for c in customers:
        if curr_load + demands[c] > capacity:
            seq.append(0)
            curr_load = 0.0
        seq.append(c)
        curr_load += demands[c]

    state.sequence = seq
    state.makespan = calc_makespan(seq, env_data)


# ── Pipeline configuration ────────────────────────────────────────


def get_initial_pipeline():
    """Return the default CVRP operator pipeline."""
    return [
        "initialization|init_clarke_wright",
        "intra_route_ls|ls_2opt",
        "perturbation|mut_random_swap",
        "inter_route_ls|ls_relocate",
        "intensification|simulated_annealing",
    ]


def get_allowed_nodes(node_ablation_level):
    """Return the exact nested CVRP operator pool for a V-Level."""
    level_4 = {
        "initialization|init_nearest_neighbor",
        "intensification|simulated_annealing",
        "intra_route_ls|ls_2opt",
        "perturbation|mut_random_swap",
    }
    level_8 = level_4 | {
        "initialization|init_clarke_wright",
        "inter_route_ls|ls_exchange",
        "inter_route_ls|ls_relocate",
        "perturbation|mut_route_split",
    }
    level_15 = level_8 | {
        "diversification|div_random_walk",
        "initialization|init_random",
        "intensification|vns_descent",
        "inter_route_ls|ls_cross",
        "intra_route_ls|ls_3opt",
        "ruin_and_recreate|ruin_radial",
        "ruin_and_recreate|ruin_random",
    }
    level_20 = level_15 | {
        "crossover|history_elite_crossover",
        "initialization|init_sweep_heuristic",
        "intensification|tabu_search_lite",
        "intra_route_ls|ls_or_opt",
        "perturbation|mut_bottleneck_route",
    }
    pools = {4: level_4, 8: level_8, 15: level_15, 20: level_20}
    try:
        level = int(node_ablation_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("CVRP node_ablation_level must be one of 4, 8, 15, or 20") from exc
    if level not in pools:
        raise ValueError(
            f"Unsupported CVRP node_ablation_level={node_ablation_level!r}; "
            "expected one of 4, 8, 15, or 20"
        )
    return set(pools[level])


# ── Domain prompt ─────────────────────────────────────────────────


def get_prompt(root_dir=None):
    """
    Read the CVRP domain knowledge prompt from disk.

    Args:
        root_dir: Project root directory. If None, auto-detected relative
                  to this file (projects/cvrp/ -> src/ -> root/).
    """
    if root_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    prompt_path = os.path.join(root_dir, "prompts", "cvrp", "domain_knowledge.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_reward_spec(env_data):
    from src.core.contracts import default_reward_spec

    return default_reward_spec(env_data)


def describe_instance(env_data):
    from src.core.contracts import default_describe_instance

    return default_describe_instance(env_data)
