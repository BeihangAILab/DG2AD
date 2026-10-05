import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """
    Perturb the priority sequence by swapping multiple random pairs.

    Swaps k pairs of items where k is between 2 and 15% of n.
    Saves the previous state for potential rollback by acceptance.
    """
    n = env_data["num_activities"]
    seq = np.array(state.sequence, dtype=np.int32).copy()
    if n < 2:
        return state
    original = seq.copy()

    md = state.metadata
    md["prev_sequence"] = seq.copy()
    md["prev_makespan"] = state.makespan

    k = max(2, int(n * random.uniform(0.02, 0.15)))
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        seq[i], seq[j] = seq[j], seq[i]

    if np.array_equal(seq, original):
        seq[0], seq[1] = seq[1], seq[0]

    state.sequence = seq
    state.makespan = calc_makespan_fn(state, env_data)
    return state
