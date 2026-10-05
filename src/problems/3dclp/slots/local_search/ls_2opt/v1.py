import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Evaluate at most three segment reversals and commit the best improvement."""
    n = env_data["num_activities"]
    original = np.asarray(state.sequence, dtype=np.int32)
    if n < 2:
        return state
    pairs = {(0, n - 1)}
    budget = min(3, n * (n - 1) // 2)
    while len(pairs) < budget:
        i, j = sorted(random.sample(range(n), 2))
        pairs.add((i, j))

    best_seq = original.copy()
    best_score = state.makespan
    for i, j in pairs:
        candidate = original.copy()
        candidate[i : j + 1] = candidate[i : j + 1][::-1]
        score = calc_makespan_fn(candidate, env_data)
        if score < best_score:
            best_seq, best_score = candidate, score

    state.sequence = best_seq
    state.makespan = best_score
    return state
