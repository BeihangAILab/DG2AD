import random
import numpy as np


def _insert(sequence, source, target):
    values = sequence.tolist()
    item = values.pop(source)
    values.insert(target, item)
    return np.asarray(values, dtype=np.int32)


def run(env_data, state, calc_makespan_fn):
    """Search swap, insertion, and reversal neighborhoods with nine evaluations."""
    n = env_data["num_activities"]
    original = np.asarray(state.sequence, dtype=np.int32)
    if n < 2:
        return state
    best_seq = original.copy()
    best_score = state.makespan
    for neighborhood in ("swap", "insert", "reverse"):
        for _ in range(3):
            i, j = sorted(random.sample(range(n), 2))
            candidate = original.copy()
            if neighborhood == "swap":
                candidate[i], candidate[j] = candidate[j], candidate[i]
            elif neighborhood == "insert":
                candidate = _insert(original, i, j)
            else:
                candidate[i : j + 1] = candidate[i : j + 1][::-1]
            score = calc_makespan_fn(candidate, env_data)
            if score < best_score:
                best_seq, best_score = candidate, score

    state.sequence = best_seq
    state.makespan = best_score
    return state
