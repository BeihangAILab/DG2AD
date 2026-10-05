import random

import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Apply a legal, deliberately non-greedy tour perturbation."""
    n = env_data["num_nodes"]
    if n < 2:
        return state

    metadata = state.metadata
    metadata["prev_sequence"] = np.asarray(state.sequence, dtype=np.int32).copy()
    metadata["prev_makespan"] = state.makespan

    sequence = list(state.sequence)
    original = sequence[:]
    move_count = max(2, int(n * random.uniform(0.02, 0.15)))
    for _ in range(move_count):
        if random.random() < 0.7:
            left = random.randint(0, n - 2)
            right = random.randint(left + 1, n - 1)
            sequence[left : right + 1] = reversed(sequence[left : right + 1])
        else:
            first, second = random.sample(range(n), 2)
            sequence[first], sequence[second] = (sequence[second], sequence[first])

    if sequence == original:
        sequence[0], sequence[1] = sequence[1], sequence[0]

    state.sequence = np.asarray(sequence, dtype=np.int32)
    state.makespan = calc_makespan_fn(state.sequence, env_data)
    return state
