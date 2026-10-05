import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Search a bounded sample of adjacent priority swaps."""
    n = env_data["num_activities"]
    original = np.asarray(state.sequence, dtype=np.int32)
    if n < 2:
        return state
    positions = list(range(n - 1))
    random.shuffle(positions)
    best_seq = original.copy()
    best_score = state.makespan
    for index in positions[:8]:
        candidate = original.copy()
        candidate[index], candidate[index + 1] = candidate[index + 1], candidate[index]
        score = calc_makespan_fn(candidate, env_data)
        if score < best_score:
            best_seq, best_score = candidate, score

    state.sequence = best_seq
    state.makespan = best_score
    return state
