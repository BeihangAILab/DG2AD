import itertools
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Prioritise large, dimension-tight items that are hard to place later."""
    n = env_data["num_activities"]
    current = np.asarray(state.sequence).ravel()
    if len(current) == n and np.isfinite(float(state.makespan)):
        return state
    sizes = np.asarray(env_data["item_sizes"], dtype=np.float64)
    bin_size = np.asarray(env_data["bin_size"], dtype=np.float64)
    container_volume = max(float(np.prod(bin_size)), 1.0)
    scores = []
    for index, dims in enumerate(sizes):
        rotations = [
            np.asarray(rot, dtype=np.float64) for rot in set(itertools.permutations(dims.tolist()))
        ]
        feasible = [rot for rot in rotations if np.all(rot <= bin_size)]
        ratios = [float(np.max(rot / bin_size)) for rot in feasible] or [10.0]
        volume_term = float(np.prod(dims)) / container_volume
        scores.append((2.0 * volume_term + min(ratios), index))
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
