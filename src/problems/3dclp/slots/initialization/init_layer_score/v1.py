import itertools
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Prioritise items with useful base coverage and shallow feasible height."""
    n = env_data["num_activities"]
    current = np.asarray(state.sequence).ravel()
    if len(current) == n and np.isfinite(float(state.makespan)):
        return state
    sizes = np.asarray(env_data["item_sizes"], dtype=np.float64)
    bin_size = np.asarray(env_data["bin_size"], dtype=np.float64)
    base_area = max(float(bin_size[0] * bin_size[2]), 1.0)
    scores = []
    for index, dims in enumerate(sizes):
        orientation_scores = []
        for rot_tuple in set(itertools.permutations(dims.tolist())):
            rot = np.asarray(rot_tuple, dtype=np.float64)
            if np.all(rot <= bin_size):
                footprint = float(rot[0] * rot[2]) / base_area
                height_penalty = float(rot[1] / max(bin_size[1], 1.0))
                orientation_scores.append(footprint - 0.2 * height_penalty)
        scores.append((max(orientation_scores) if orientation_scores else -1.0, index))
    scores.sort(reverse=True)
    seq = np.asarray([index for _, index in scores], dtype=np.int32)
    score = calc_makespan_fn(seq, env_data)
    state.sequence = seq
    state.makespan = score
    state.metadata = {
        "temp": 5.0,
        "best_score": score,
        "best_seq": seq.copy(),
        "prev_makespan": score,
        "prev_sequence": seq.copy(),
        "no_improve": 0,
    }
    return state
