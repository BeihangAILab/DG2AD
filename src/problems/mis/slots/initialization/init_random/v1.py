import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Build a maximal independent set from a uniformly random priority."""
    n = env_data["num_vertices"]
    current = np.asarray(state.sequence).ravel()
    if len(current) == n and np.isfinite(float(state.makespan)):
        return state
    adj = env_data["adj"]
    order = list(range(n))
    random.shuffle(order)
    seq = np.zeros(n, dtype=np.int32)
    blocked = np.zeros(n, dtype=np.int32)
    for vertex in order:
        if blocked[vertex] == 0:
            seq[vertex] = 1
            for neighbor in adj[vertex]:
                blocked[neighbor] = 1

    score = calc_makespan_fn(seq, env_data)
    state.sequence = seq
    state.makespan = score
    state.metadata = {
        "temp": 5.0,
        "best_score": score,
        "best_seq": seq.copy(),
        "prev_makespan": score,
        "prev_sequence": seq.copy(),
        "no_improve": 0,
    }
    return state
