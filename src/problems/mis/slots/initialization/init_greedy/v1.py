import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    n = env_data["num_vertices"]
    current = np.asarray(state.sequence).ravel()
    if len(current) == n and np.isfinite(float(state.makespan)):
        return state
    adj = env_data["adj"]
    degrees = np.array([len(adj[i]) for i in range(n)], dtype=np.int32)
    order = list(range(n))
    random.shuffle(order)
    order.sort(key=lambda i: degrees[i])

    seq = np.zeros(n, dtype=np.int32)
    blocked = np.zeros(n, dtype=np.int32)
    for v in order:
        if blocked[v] == 0:
            seq[v] = 1
            for u in adj[v]:
                blocked[u] = 1

    state.sequence = seq
    state.makespan = calc_makespan_fn(state, env_data)
    state.metadata = {
        "temp": 5.0,
        "best_score": state.makespan,
        "best_seq": seq.copy(),
        "prev_makespan": state.makespan,
        "prev_sequence": seq.copy(),
        "no_improve": 0,
    }
    return state
