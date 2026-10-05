import random
import numpy as np


def _relocate(sequence, start, length, target):
    values = sequence.tolist()
    block = values[start : start + length]
    del values[start : start + length]
    target = min(target, len(values))
    values[target:target] = block
    return np.asarray(values, dtype=np.int32)


def run(env_data, state, calc_makespan_fn):
    """Evaluate at most eight relocations of short contiguous item blocks."""
    n = env_data["num_activities"]
    original = np.asarray(state.sequence, dtype=np.int32)
    if n < 3:
        return state
    best_seq = original.copy()
    best_score = state.makespan
    moves = set()
    attempts = 0
    while len(moves) < min(8, n) and attempts < 40:
        attempts += 1
        length = random.randint(2, min(5, n - 1))
        start = random.randint(0, n - length)
        target = random.randint(0, n - length)
        if target != start:
            moves.add((start, length, target))
    for start, length, target in moves:
        candidate = _relocate(original, start, length, target)
        if np.array_equal(candidate, original):
            continue
        score = calc_makespan_fn(candidate, env_data)
        if score < best_score:
            best_seq, best_score = candidate, score

    state.sequence = best_seq
    state.makespan = best_score
    return state
