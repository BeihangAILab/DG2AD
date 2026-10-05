import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Reverse a random block of at least two priority items."""
    n = env_data["num_activities"]
    if n < 2:
        return state
    original = np.asarray(state.sequence, dtype=np.int32).copy()
    state.metadata["prev_sequence"] = original.copy()
    state.metadata["prev_makespan"] = state.makespan
    max_length = min(n, max(2, int(round(n * 0.2))))
    length = random.randint(2, max_length)
    start = random.randint(0, n - length)
    candidate = original.copy()
    candidate[start : start + length] = candidate[start : start + length][::-1]
    state.sequence = candidate
    state.makespan = calc_makespan_fn(candidate, env_data)
    return state
