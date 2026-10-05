import random
import numpy as np


def _insert(values, source, target):
    item = values.pop(source)
    values.insert(target, item)


def run(env_data, state, calc_makespan_fn):
    """Apply a short unfiltered walk of swap and insertion moves."""
    n = env_data["num_activities"]
    if n < 2:
        return state
    original = np.asarray(state.sequence, dtype=np.int32).copy()
    state.metadata["prev_sequence"] = original.copy()
    state.metadata["prev_makespan"] = state.makespan
    values = original.tolist()
    for _ in range(random.randint(3, 6)):
        i, j = random.sample(range(n), 2)
        if random.random() < 0.5:
            values[i], values[j] = values[j], values[i]
        else:
            _insert(values, i, j)
    candidate = np.asarray(values, dtype=np.int32)
    if np.array_equal(candidate, original):
        candidate[0], candidate[1] = candidate[1], candidate[0]
    state.sequence = candidate
    state.makespan = calc_makespan_fn(candidate, env_data)
    return state
