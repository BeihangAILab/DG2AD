"""
3DBPP domain evaluator — 3D Bin Packing Problem.

Exposes the DGA2D domain protocol:
  load_instance_group, get_reward_spec, describe_instance, calc_makespan,
  get_initial_pipeline, initialize_state, get_prompt

The 3DBPP uses a priority-based sequence encoding:
  sequence[i] = item_id to pack at step i.
Items are packed in order using the Extreme Points (EP) First-Fit
heuristic decoder (serial_sgs).

Objective: bins + (1.0 - last_bin_utilization) * 0.1  (minimisation)
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
    return str(problem_data_dir("3dbbp", os.path.join(os.path.dirname(__file__), "instances")))


def _parse_mpv_txt(filepath):
    """
    Parse a 3DBPP MPV .txt instance file.

    Format:
      Line 1: W H D (bin dimensions)
      Line 2: n (number of items)
      Lines 3..2+n: w h d (item dimensions, one per line)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    W, H, D = map(int, lines[0].split())
    n = int(lines[1].strip())
    item_sizes_list = []
    for i in range(2, 2 + n):
        item_sizes_list.append(list(map(int, lines[i].split())))
    item_sizes = np.array(item_sizes_list, dtype=np.int32)
    return W, H, D, n, item_sizes


def _load_bks_db():
    """Load the BKS (best-known solution) database."""
    db_path = os.path.join(_get_instances_dir(), "3dbpp_instances_db.json")
    if os.path.exists(db_path):
        with open(db_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _lookup_bks(instance_name):
    """Look up the BKS upper bound for a given instance name."""
    db = _load_bks_db()
    for entry in db:
        if entry["name"] == instance_name:
            return entry["ub"]
    return None


def _get_all_instances():
    """Discover all MPV .txt instances in the instances directory."""
    mpv_dir = os.path.join(_get_instances_dir(), "MPV")
    instances = []
    if os.path.isdir(mpv_dir):
        for fname in sorted(os.listdir(mpv_dir)):
            if fname.endswith(".txt"):
                name = fname.replace(".txt", "")
                filepath = os.path.join(mpv_dir, fname)
                instances.append((name, filepath))
    return instances


# ── Data loading ─────────────────────────────────────────────────────


def load_instance_group(group):
    """Load every instance in the requested dataset group."""
    all_instances = _get_all_instances()

    if group.startswith("MPV_"):
        size_suffix = group.split("_n")[1]
        group_instances = [
            (name, fp) for name, fp in all_instances if name.startswith(f"instance_n{size_suffix}_")
        ]
    else:
        # Fallback: filter by group contained in name
        group_instances = [(name, fp) for name, fp in all_instances if group in name]

    if not group_instances:
        return []

    results = []
    for name, filepath in group_instances:
        results.append(_inst_to_env(name, filepath))
    return results


def _inst_to_env(name, filepath):
    """Convert a raw 3DBPP instance to the standard env_data format."""
    W, H, D, n, item_sizes = _parse_mpv_txt(filepath)
    item_volumes = item_sizes[:, 0] * item_sizes[:, 1] * item_sizes[:, 2]
    ub = _lookup_bks(name)
    if ub is None:
        ub = n  # worst case: one item per bin

    return {
        "name": name,
        "bin_size": np.array([W, H, D], dtype=np.int32),
        "num_activities": n,
        "item_sizes": item_sizes,
        "item_volumes": item_volumes.astype(np.int32),
        "upper_bound": float(ub),
        "lower_bound": float(ub),
    }


# ── EP First-Fit decoder (ported from LLM-Evolution-Scheduler) ──────


@njit
def _check_overlap(x1, y1, z1, w1, h1, d1, x2, y2, z2, w2, h2, d2):
    """Check if two 3D boxes overlap."""
    return (
        max(x1, x2) < min(x1 + w1, x2 + w2)
        and max(y1, y2) < min(y1 + h1, y2 + h2)
        and max(z1, z2) < min(z1 + d1, z2 + d2)
    )


@njit
def serial_sgs(priority_list, num_activities, bin_size, item_sizes, item_volumes):
    """
    Extreme Points (EP) First-Fit 3D bin packing decoder.

    Packs items in the order given by priority_list into bins.
    Each bin is packed using an EP heuristic: candidate points at
    (x+w, y, z), (x, y+h, z), (x, y, z+d), sorted BLF (z then y then x).

    Returns:
        num_bins_used (float), last_bin_utilization (float in [0, 1])
    """
    W = bin_size[0]
    H = bin_size[1]
    D = bin_size[2]
    bin_volume = W * H * D

    placed_x = np.zeros(num_activities, dtype=np.int32)
    placed_y = np.zeros(num_activities, dtype=np.int32)
    placed_z = np.zeros(num_activities, dtype=np.int32)
    placed_w = np.zeros(num_activities, dtype=np.int32)
    placed_h = np.zeros(num_activities, dtype=np.int32)
    placed_d = np.zeros(num_activities, dtype=np.int32)
    placed_bin = -np.ones(num_activities, dtype=np.int32)

    num_bins_used = 0
    total_placed = 0

    for act_idx in range(len(priority_list)):
        act = priority_list[act_idx]
        wi = item_sizes[act, 0]
        hi = item_sizes[act, 1]
        di = item_sizes[act, 2]

        # Fixed orientation (no rotation)
        orientations = [(wi, hi, di)]

        placed_ok = False

        # Try each existing bin (First-Fit)
        for b in range(num_bins_used):
            # Gather EP candidate points from items in this bin
            ep_x = [0]
            ep_y = [0]
            ep_z = [0]

            for j in range(total_placed):
                if placed_bin[j] == b:
                    xj = placed_x[j]
                    yj = placed_y[j]
                    zj = placed_z[j]
                    wj = placed_w[j]
                    hj = placed_h[j]
                    dj = placed_d[j]

                    ep_x.append(xj + wj)
                    ep_y.append(yj)
                    ep_z.append(zj)

                    ep_x.append(xj)
                    ep_y.append(yj + hj)
                    ep_z.append(zj)

                    ep_x.append(xj)
                    ep_y.append(yj)
                    ep_z.append(zj + dj)

            # Sort EP points by z, then y, then x (BLF order)
            num_eps = len(ep_x)
            for step in range(num_eps):
                for idx in range(num_eps - step - 1):
                    swap_flag = False
                    if ep_z[idx] > ep_z[idx + 1]:
                        swap_flag = True
                    elif ep_z[idx] == ep_z[idx + 1]:
                        if ep_y[idx] > ep_y[idx + 1]:
                            swap_flag = True
                        elif ep_y[idx] == ep_y[idx + 1]:
                            if ep_x[idx] > ep_x[idx + 1]:
                                swap_flag = True

                    if swap_flag:
                        tx = ep_x[idx]
                        ep_x[idx] = ep_x[idx + 1]
                        ep_x[idx + 1] = tx
                        ty = ep_y[idx]
                        ep_y[idx] = ep_y[idx + 1]
                        ep_y[idx + 1] = ty
                        tz = ep_z[idx]
                        ep_z[idx] = ep_z[idx + 1]
                        ep_z[idx + 1] = tz

            # Try each EP point
            best_x = best_y = best_z = -1
            best_w = best_h = best_d = -1
            found_ep = False

            for idx in range(num_eps):
                ex = ep_x[idx]
                ey = ep_y[idx]
                ez = ep_z[idx]

                for rot in orientations:
                    rw, rh, rd = rot

                    # Boundary check
                    if ex + rw > W or ey + rh > H or ez + rd > D:
                        continue

                    # Overlap check
                    has_overlap = False
                    for j in range(total_placed):
                        if placed_bin[j] == b:
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
                        best_x, best_y, best_z = ex, ey, ez
                        best_w, best_h, best_d = rw, rh, rd
                        found_ep = True
                        break
                if found_ep:
                    break

            if found_ep:
                # Place item in this bin
                placed_x[total_placed] = best_x
                placed_y[total_placed] = best_y
                placed_z[total_placed] = best_z
                placed_w[total_placed] = best_w
                placed_h[total_placed] = best_h
                placed_d[total_placed] = best_d
                placed_bin[total_placed] = b
                total_placed += 1
                placed_ok = True
                break

        if not placed_ok:
            # Open a new bin at (0, 0, 0)
            b = num_bins_used
            placed_x[total_placed] = 0
            placed_y[total_placed] = 0
            placed_z[total_placed] = 0
            placed_w[total_placed] = wi
            placed_h[total_placed] = hi
            placed_d[total_placed] = di
            placed_bin[total_placed] = b
            total_placed += 1
            num_bins_used += 1

    # Compute last bin utilization
    last_bin_idx = num_bins_used - 1
    last_bin_volume = 0
    for j in range(total_placed):
        if placed_bin[j] == last_bin_idx:
            last_bin_volume += placed_w[j] * placed_h[j] * placed_d[j]

    last_bin_util = 0.0
    if bin_volume > 0:
        last_bin_util = last_bin_volume / bin_volume

    return float(num_bins_used), last_bin_util


# ── Solution evaluation (makespan = bins + utilization penalty) ─────


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
    Compute 3DBPP objective from a priority sequence.

    The engine passes either a numpy array (the sequence) or a SolutionState
    object with .sequence attribute.

    Objective = bins + (1.0 - last_bin_utilization) * 0.1

    Returns a float to minimise.
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

    bins, last_util = serial_sgs(
        arr,
        n,
        env_data["bin_size"],
        env_data["item_sizes"],
        env_data["item_volumes"],
    )
    return bins + (1.0 - last_util) * 0.1


# ── State initialisation ─────────────────────────────────────────────


def initialize_state(env_data, state):
    """
    Ensure state has a feasible 3DBPP solution if uninitialised.

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
    """Return the default 3DBPP operator pipeline."""
    return [
        "initialization|init_volume_desc",
        "perturbation|mut_swap",
        "local_search|ls_2opt",
        "acceptance|simulated_annealing",
    ]


# ── Domain prompt ────────────────────────────────────────────────────


def get_prompt(root_dir=None):
    """
    Read the 3DBPP domain knowledge prompt from disk.

    Args:
        root_dir: Project root directory. If None, auto-detected relative
                  to this file (problems/3dbbp/ -> src/ -> root/).
    """
    if root_dir is None:
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    prompt_path = os.path.join(root_dir, "prompts", "3dbbp", "domain_knowledge.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_reward_spec(env_data):
    from src.core.contracts import default_reward_spec

    return default_reward_spec(env_data)


def describe_instance(env_data):
    from src.core.contracts import default_describe_instance

    return default_describe_instance(env_data)
