import random
import numpy as np


def _insert(sequence, source, target):
    values = sequence.tolist()
    item = values.pop(source)
    values.insert(target, item)
    return np.asarray(values, dtype=np.int32)


def run(env_data, state, calc_makespan_fn):
    """Evaluate at most ten item-insertion moves."""
    n = env_data["num_activities"]
    original = np.asarray(state.sequence, dtype=np.int32)
    if n < 2:
        return state
    moves = set()
    budget = min(10, n * (n - 1))
    while len(moves) < budget:
        source, target = random.sample(range(n), 2)
        moves.add((source, target))

    best_seq = original.copy()
    best_score = state.makespan
    for source, target in moves:
        candidate = _insert(original, source, target)
        score = calc_makespan_fn(candidate, env_data)
        if score < best_score:
            best_seq, best_score = candidate, score

    state.sequence = best_seq
    state.makespan = best_score
    return state
