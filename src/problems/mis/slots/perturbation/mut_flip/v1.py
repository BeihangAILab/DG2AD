import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    n = env_data["num_vertices"]
    adj = env_data["adj"]
    seq = np.array(state.sequence, dtype=np.int32).copy()
    original = seq.copy()
    state.metadata["prev_sequence"] = seq.copy()
    state.metadata["prev_makespan"] = state.makespan

    k = max(1, int(n * random.uniform(0.05, 0.15)))
    for idx in random.sample(range(n), k):
        seq[idx] = 1 - seq[idx]

    degrees = np.array([len(adj[i]) for i in range(n)], dtype=np.int32)
    ones = [i for i in range(n) if seq[i] == 1]
    random.shuffle(ones)
    for v in ones:
        if seq[v] == 0:
            continue
        for u in adj[v]:
            if u > v and seq[u] == 1:
                if degrees[v] <= degrees[u]:
                    seq[u] = 0
                else:
                    seq[v] = 0
                    break

    if np.array_equal(seq, original) and n > 0:
        selected = np.flatnonzero(seq == 1)
        if len(selected):
            seq[random.choice(selected)] = 0
        else:
            seq[random.randrange(n)] = 1

    state.sequence = seq
    state.makespan = calc_makespan_fn(state, env_data)
    return state
