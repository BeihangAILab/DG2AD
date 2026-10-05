import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Remove a subset of priorities and reinsert large items at new positions."""
    n = env_data["num_activities"]
    if n < 2:
        return state
    original = np.asarray(state.sequence, dtype=np.int32).copy()
    state.metadata["prev_sequence"] = original.copy()
    state.metadata["prev_makespan"] = state.makespan
    count = min(n - 1, max(2, int(round(n * 0.2)))) if n > 2 else 1
    positions = set(random.sample(range(n), count))
    removed = [int(original[index]) for index in sorted(positions)]
    remaining = [int(item) for index, item in enumerate(original) if index not in positions]
    volumes = env_data["item_volumes"]
    removed.sort(key=lambda item: float(volumes[item]), reverse=True)
    for item in removed:
        target = random.randint(0, len(remaining))
        remaining.insert(target, item)
    candidate = np.asarray(remaining, dtype=np.int32)
    if np.array_equal(candidate, original):
        candidate[0], candidate[1] = candidate[1], candidate[0]
    state.sequence = candidate
    state.makespan = calc_makespan_fn(candidate, env_data)
    return state
