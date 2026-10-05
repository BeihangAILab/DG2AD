import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Shuffle 15-30 percent of positions while preserving every item."""
    n = env_data["num_activities"]
    if n < 2:
        return state
    original = np.asarray(state.sequence, dtype=np.int32).copy()
    state.metadata["prev_sequence"] = original.copy()
    state.metadata["prev_makespan"] = state.makespan
    count = min(n, max(2, int(round(n * random.uniform(0.15, 0.30)))))
    positions = sorted(random.sample(range(n), count))
    values = original[positions].tolist()
    random.shuffle(values)
    if values == original[positions].tolist():
        values = values[1:] + values[:1]
    candidate = original.copy()
    candidate[positions] = np.asarray(values, dtype=np.int32)
    state.sequence = candidate
    state.makespan = calc_makespan_fn(candidate, env_data)
    return state
