import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Randomly scramble a bounded contiguous block, forcing a new permutation."""
    n = env_data["num_activities"]
    if n < 2:
        return state
    original = np.asarray(state.sequence, dtype=np.int32).copy()
    state.metadata["prev_sequence"] = original.copy()
    state.metadata["prev_makespan"] = state.makespan
    max_length = min(n, max(3, int(round(n * 0.15)))) if n >= 3 else 2
    length = random.randint(2, max_length)
    start = random.randint(0, n - length)
    candidate = original.copy()
    block = candidate[start : start + length].tolist()
    random.shuffle(block)
    if block == candidate[start : start + length].tolist():
        block.reverse()
    candidate[start : start + length] = np.asarray(block, dtype=np.int32)
    state.sequence = candidate
    state.makespan = calc_makespan_fn(candidate, env_data)
    return state
