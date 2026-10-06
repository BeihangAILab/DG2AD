"""
3DCLP domain evaluator — 3D Container Loading Problem.

Exposes the DGA2D domain protocol:
  load_instance_group, get_reward_spec, describe_instance, calc_makespan,
  get_initial_pipeline, initialize_state, get_prompt

The 3DCLP uses a priority-based sequence encoding:
  sequence[i] = item_id to pack at step i.
Items are packed in order into a SINGLE container using the
Extreme Points (EP) First-Fit heuristic decoder with 6 rotations.
Items that cannot fit are skipped.

WARNING — 3DCLP is a MAXIMISATION problem (maximise fill rate):
  The engine minimises, so calc_makespan returns ** 100.0 - fill_rate_pct **.
  upper_bound = 100.0 - BKS_fill_rate_pct  (also in minimisation space).
  GAP is computed via: (score - UB) / UB × 100
  which gives the % deviation from optimal in the minimisation space.
  0% = optimal, positive = suboptimal.
"""

import json
import os
import random
import numpy as np

from src.core.configuration import problem_data_dir

INVALID_SCORE = 99999.0

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


def _get_instances_dir():
    """Return the absolute path to the instances directory."""
    return str(problem_data_dir("3dclp", os.path.join(os.path.dirname(__file__), "instances")))


def _parse_br_json(filepath):
    """
    Parse a 3DCLP BR .json instance file.

    Format:
      {
        "Objects": [{"Length": W, "Height": H, "Depth": D}],
        "Items": [{"Length": w, "Height": h, "Depth": d, "Demand": count}, ...]
      }

    Returns:
        W, H, D (container dims), n (total items after demand expansion),
        item_sizes (np.array shape (n, 3))
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    obj = data["Objects"][0]
    W = int(obj["Length"])
    H = int(obj["Height"])
    D = int(obj["Depth"])

    item_sizes_list = []
    orientation_allowed_list = []
    for it in data["Items"]:
        w = int(it["Length"])
        h = int(it["Height"])
        d = int(it["Depth"])
        demand = int(it.get("Demand", 1))
        allowed = [
            int(it.get("C1_Length", 1)),
            int(it.get("C1_Height", 1)),
            int(it.get("C1_Depth", 1)),
        ]
        for _ in range(demand):
            item_sizes_list.append([w, h, d])
            orientation_allowed_list.append(allowed)

    item_sizes = np.array(item_sizes_list, dtype=np.int32)
    orientation_allowed = np.array(orientation_allowed_list, dtype=np.int8)
    n = len(item_sizes)
    return W, H, D, n, item_sizes, orientation_allowed


def _load_bks_db():
    """Load the BKS (best-known solution) database for 3DCLP."""
    db_path = os.path.join(_get_instances_dir(), "3dclp_bks.json")
    if os.path.exists(db_path):
        with open(db_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _lookup_bks(instance_name):
    """
    Look up the BKS fill_rate for a given instance name.

    Returns fill_rate as a percentage (0-100), or None if not found.
    """
    db = _load_bks_db()
    for entry in db:
        if entry.get("name") == instance_name:
            fr = entry.get("fill_rate", None)
            if fr is not None:
                return float(fr) * 100.0  # Convert 0-1 to 0-100 percentage
    return None


def _get_all_instances():
    """Discover all BR .json instances in the instances directory."""
    inst_dir = os.path.join(_get_instances_dir(), "3dclp_instances")
    instances = []
    if os.path.isdir(inst_dir):
        for fname in sorted(os.listdir(inst_dir)):
            if fname.endswith(".json"):
                name = fname.replace(".json", "")
                filepath = os.path.join(inst_dir, fname)
                instances.append((name, filepath))
    return instances


# ── EP First-Fit decoder for single container with 6 rotations ──────


@njit
def _check_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
    """Check if two 3D boxes overlap."""
    return (
        max(x1, x2) < min(x1 + w1, x2 + w2)
        and max(y1, y2) < min(y1 + h1, y2 + h2)
        and max(z1, z2) < min(z1 + d1, z2 + d2)
    )


@njit
def _single_container_pack(
    priority_list, num_activities, bin_size, item_sizes, orientation_allowed
):
    """
    Single container 3DCLP: EP-FF + 6 rotations.

    Packs items in priority order into ONE container. Items that cannot
    fit in any orientation at any EP point are skipped.

    Args:
        priority_list: array of item indices in packing order
        num_activities: total number of items
        bin_size: np.array([W, H, D])
        item_sizes: np.array shape (N, 3) of (w, h, d)

    Returns:
        fill_rate (float in [0, 1])
    """
    W = bin_size[0]
    H = bin_size[1]
    D = bin_size[2]
    container_vol = W * H * D

    placed_x = np.zeros(num_activities, dtype=np.int32)
    placed_y = np.zeros(num_activities, dtype=np.int32)
    placed_z = np.zeros(num_activities, dtype=np.int32)
    placed_w = np.zeros(num_activities, dtype=np.int32)
    placed_h = np.zeros(num_activities, dtype=np.int32)
    placed_d = np.zeros(num_activities, dtype=np.int32)
    placed_flag = np.zeros(num_activities, dtype=np.int8)

    total_placed = 0
    packed_volume = 0
    # Maintain unique usable EPs in the original (z, y, x) order.
    # Duplicate coordinates cannot change the first feasible placement.
    points = [(0, 0, 0)]

    for act_idx in range(len(priority_list)):
        act = priority_list[act_idx]
        wi = item_sizes[act, 0]
        hi = item_sizes[act, 1]
        di = item_sizes[act, 2]

        # All 6 axis-aligned rotation orientations
        orientations = [
            (wi, hi, di),
            (wi, di, hi),
            (hi, wi, di),
            (hi, di, wi),
            (di, wi, hi),
            (di, hi, wi),
        ]

        num_eps = len(points)

        # Try each EP point with each rotation
        placed_ok = False
        for idx in range(num_eps):
            ez, ey, ex = points[idx]

            for rot_idx in range(6):
                # BR C1_* flags specify whether the corresponding original
                # item edge may be used as the vertical dimension.
                if rot_idx == 0 or rot_idx == 5:
                    vertical_axis = 1  # original Height
                elif rot_idx == 1 or rot_idx == 3:
                    vertical_axis = 2  # original Depth
                else:
                    vertical_axis = 0  # original Length
                if orientation_allowed[act, vertical_axis] == 0:
                    continue

                rw = orientations[rot_idx][0]
                rh = orientations[rot_idx][1]
                rd = orientations[rot_idx][2]

                # Boundary check
                if ex + rw > W or ey + rh > H or ez + rd > D:
                    continue

                # Overlap check against all placed items
                has_overlap = False
                for j in range(total_placed):
                    if placed_flag[j] == 1:
                        if _check_overlap(
                            ex,
                            ey,
                            ez,
                            rw,
                            rh,
                            rd,
                            placed_x[j],
                            placed_y[j],
                            placed_z[j],
                            placed_w[j],
                            placed_h[j],
                            placed_d[j],
                        ):
                            has_overlap = True
                            break

                if not has_overlap:
                    placed_x[total_placed] = ex
                    placed_y[total_placed] = ey
                    placed_z[total_placed] = ez
                    placed_w[total_placed] = rw
                    placed_h[total_placed] = rh
                    placed_d[total_placed] = rd
                    placed_flag[total_placed] = 1
                    total_placed += 1
                    packed_volume += rw * rh * rd
                    placed_ok = True
                    # An EP inside the newly occupied half-open box can never
                    # start another positive-size box, in any orientation.
                    points = [p for p in points if not (
                        ex <= p[2] < ex + rw and ey <= p[1] < ey + rh
                        and ez <= p[0] < ez + rd
                    )]
                    for point in ((ez, ey, ex + rw), (ez, ey + rh, ex), (ez + rd, ey, ex)):
                        pz, py, px = point
                        if px >= W or py >= H or pz >= D:
                            continue
                        # Binary insertion preserves BLF ordering exactly.
                        lo, hi = 0, len(points)
                        while lo < hi:
                            mid = (lo + hi) // 2
                            if points[mid] < point:
                                lo = mid + 1
                            else:
                                hi = mid
                        if lo == len(points) or points[lo] != point:
                            points.insert(lo, point)
                    break

            if placed_ok:
                break

    fill_rate = packed_volume / container_vol if container_vol > 0 else 0.0
    return fill_rate


# ── Data loading ─────────────────────────────────────────────────────


def _inst_to_env(name, filepath):
    """Convert a raw 3DCLP instance to the standard env_data format.

    BKS fill_rate (0-100) is converted to minimisation space:
      upper_bound = 100.0 - fill_rate_pct_bks
    This matches calc_makespan's return value (also 100.0 - fill_rate_pct),
    so reward normalization works correctly.
    """
    W, H, D, n, item_sizes, orientation_allowed = _parse_br_json(filepath)
    item_volumes = (item_sizes[:, 0] * item_sizes[:, 1] * item_sizes[:, 2]).astype(np.int32)
    bks_fill_rate = _lookup_bks(name)
    container_volume = float(W) * float(H) * float(D)
    total_item_volume = float(np.sum(item_volumes, dtype=np.float64))
    fill_upper_bound = (
        min(100.0, total_item_volume / container_volume * 100.0) if container_volume > 0.0 else 0.0
    )

    # The engine minimises residual fill. This is the corresponding valid
    # theoretical lower bound in the transformed objective space.
    upper_bound = 100.0 - fill_upper_bound

    return {
        "name": name,
        "bin_size": np.array([W, H, D], dtype=np.int32),
        "num_activities": n,
        "item_sizes": item_sizes,
        "orientation_allowed": orientation_allowed,
        "item_volumes": item_volumes,
        "fill_upper_bound": fill_upper_bound,
        "bks_fill_rate": (float(bks_fill_rate) if bks_fill_rate is not None else None),
        "upper_bound": upper_bound,
        "lower_bound": upper_bound,
    }


def load_instance_group(group):
    """Load every instance in the requested dataset group."""
    all_instances = _get_all_instances()

    # Filter by group prefix.
    # Instance names are like "BR1_5", "BR10_15", etc.
    # group can be "BR" (all), "BR1" (BR1 class), or "BR10" (BR10 class).
    group_upper = group.upper()
    if group_upper == "BR":
        # Match all BR instances
        group_instances = [
            (name, fp) for name, fp in all_instances if name.upper().startswith("BR")
        ]
    else:
        # Match specific class: e.g. "BR1" matches "BR1_5", "BR1_15", etc.
        group_instances = [
            (name, fp) for name, fp in all_instances if name.upper().startswith(f"{group_upper}_")
        ]

    if not group_instances:
        return []

    results = []
    for name, filepath in group_instances:
        results.append(_inst_to_env(name, filepath))
    return results


# ── Solution evaluation (makespan = 100 - fill_rate_pct) ────────────


def _seq_to_array(sequence, n):
    """Convert various sequence representations to a numpy array."""
    if sequence is None:
        return None
    raw = np.asarray(sequence).ravel()
    if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
        return None
    arr = raw.astype(np.int32)

    if len(arr) != n:
        return None
    # Validate item IDs
    if arr.min() < 0 or arr.max() >= n:
        return None
    # Check for duplicates
    if len(set(arr)) != n:
        return None
    return arr


def calc_makespan(sequence_or_state, env_data):
    """
    Compute 3DCLP objective from a priority sequence.

    Objective = 100.0 - fill_rate_pct

    Returns a float to minimise. Lower is better.
    Returns 99999.0 for invalid/infeasible sequences.
    """
    n = env_data["num_activities"]

    # Polymorphic: handle both SolutionState and raw sequence
    if hasattr(sequence_or_state, "sequence"):
        seq = sequence_or_state.sequence
    else:
        seq = sequence_or_state

    try:
        arr = _seq_to_array(seq, n)
    except (TypeError, ValueError, OverflowError):
        return INVALID_SCORE
    if arr is None:
        return INVALID_SCORE

    fill_rate = _single_container_pack(
        arr,
        n,
        env_data["bin_size"],
        env_data["item_sizes"],
        env_data.get(
            "orientation_allowed",
            np.ones((n, 3), dtype=np.int8),
        ),
    )
    fill_rate_pct = fill_rate * 100.0
    return 100.0 - fill_rate_pct


# ── State initialisation ─────────────────────────────────────────────


def initialize_state(env_data, state):
    """
    Ensure state has a feasible 3DCLP solution if uninitialised.

    Creates a random priority sequence of all items.
    """
    if state.sequence is not None and len(state.sequence) > 0 and state.makespan < float("inf"):
        return  # Already valid

    n = env_data["num_activities"]
    seq = list(range(n))
    random.shuffle(seq)
    state.sequence = np.array(seq, dtype=np.int32)
    state.makespan = calc_makespan(state, env_data)


# ── Pipeline configuration ───────────────────────────────────────────


def get_initial_pipeline():
    """Return the default 3DCLP operator pipeline."""
    return [
        "initialization|init_volume_desc",
        "perturbation|mut_swap",
        "local_search|ls_2opt",
        "acceptance|simulated_annealing",
    ]


# ── Domain prompt ────────────────────────────────────────────────────


def get_allowed_nodes(node_ablation_level):
    """Return the exact nested 3DCLP operator pool for a V-Level."""
    level_4 = {
        "acceptance|simulated_annealing",
        "initialization|init_volume_desc",
        "local_search|ls_2opt",
        "perturbation|mut_swap",
    }
    level_8 = level_4 | {
        "initialization|init_random",
        "local_search|ls_adjacent_swap",
        "local_search|ls_insert",
        "perturbation|mut_insert",
    }
    level_15 = level_8 | {
        "diversification|div_random_walk",
        "initialization|init_dimension_score",
        "initialization|init_layer_score",
        "intensification|vns_descent",
        "local_search|ls_block_relocate",
        "perturbation|mut_block_reverse",
        "perturbation|mut_scramble",
    }
    level_20 = level_15 | {
        "acceptance|late_acceptance",
        "diversification|div_partial_shuffle",
        "history_mining|history_elite_crossover",
        "intensification|iterated_greedy",
        "ruin_and_recreate|ruin_reinsert",
    }
    pools = {4: level_4, 8: level_8, 15: level_15, 20: level_20}
    try:
        level = int(node_ablation_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("3DCLP node_ablation_level must be one of 4, 8, 15, or 20") from exc
    if level not in pools:
        raise ValueError(
            f"Unsupported 3DCLP node_ablation_level={node_ablation_level!r}; "
            "expected one of 4, 8, 15, or 20"
        )
    return set(pools[level])


def get_prompt(root_dir=None):
    """
    Read the 3DCLP domain knowledge prompt from disk.

    Args:
        root_dir: Project root directory. If None, auto-detected relative
                  to this file (problems/3dclp/ -> src/ -> root/).
    """
    if root_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    prompt_path = os.path.join(root_dir, "prompts", "3dclp", "domain_knowledge.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_reward_spec(env_data):
    from src.core.contracts import RewardSpec

    # calc_makespan returns 100-fill. These residual-space values are exactly
    # equivalent to a signed, maximisation gap in fill-rate space.
    fill_reference = float(env_data["fill_upper_bound"])
    return RewardSpec(
        direction="min",
        reference_value=100.0 - fill_reference,
        normalization_scale=max(abs(fill_reference), 1.0),
    )


def describe_instance(env_data):
    from src.core.contracts import default_describe_instance

    return default_describe_instance(env_data)
