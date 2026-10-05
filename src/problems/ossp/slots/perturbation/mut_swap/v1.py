import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """
    Perturb the OSSP operation sequence via multiple random moves.

    Applies a mix of swap, insert, and reverse moves on k random
    positions (k = 3% to 15% of total operations).

    Saves the previous state for potential rollback by acceptance.
    """
    n = env_data["total_ops"]

    md = state.metadata
    original = np.array(state.sequence, dtype=np.int32).copy()
    md["prev_sequence"] = original.copy()
    md["prev_makespan"] = state.makespan

    seq = np.array(state.sequence, dtype=np.int32).copy()
    k = max(2, int(n * random.uniform(0.03, 0.15)))

    for _ in range(k):
        choice = random.random()
        if choice < 0.5:
            # Swap two random positions
            i, j = random.sample(range(n), 2)
            seq[i], seq[j] = seq[j], seq[i]
        elif choice < 0.8:
            # Remove element at i, insert at j
            i = random.randint(0, n - 1)
            val = seq[i]
            seq = np.delete(seq, i)
            j = random.randint(0, n - 1)
            seq = np.insert(seq, j, val)
        else:
            # Reverse a small segment
            i = random.randint(0, n - 2)
            length = random.randint(2, min(5, n - i))
            seq[i : i + length] = seq[i : i + length][::-1]

    if n >= 2 and np.array_equal(seq, original):
        seq[0], seq[1] = seq[1], seq[0]

    state.sequence = seq
    state.makespan = calc_makespan_fn(state, env_data)
    return state
