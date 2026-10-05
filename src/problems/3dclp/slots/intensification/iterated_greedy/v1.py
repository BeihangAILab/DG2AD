import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Evaluate four volume-guided destroy-and-reinsert candidates."""
    n = env_data["num_activities"]
    original = np.asarray(state.sequence, dtype=np.int32)
    if n < 3:
        return state
    volumes = env_data["item_volumes"]
    best_seq = original.copy()
    best_score = state.makespan
    count = min(n - 1, max(2, int(round(n * 0.1))))

    for _ in range(4):
        positions = set(random.sample(range(n), count))
        removed = [int(original[index]) for index in sorted(positions)]
        remaining = [int(item) for index, item in enumerate(original) if index not in positions]
        removed.sort(key=lambda item: float(volumes[item]), reverse=True)
        for item in removed:
            window = max(1, len(remaining) // 3)
            remaining.insert(random.randint(0, window), item)
        candidate = np.asarray(remaining, dtype=np.int32)
        score = calc_makespan_fn(candidate, env_data)
        if score < best_score:
            best_seq, best_score = candidate, score

    state.sequence = best_seq
    state.makespan = best_score
    return state
