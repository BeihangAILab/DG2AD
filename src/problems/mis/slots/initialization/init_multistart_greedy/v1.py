import random
import numpy as np


def _construct(n, adj, degrees):
    order = list(range(n))
    random.shuffle(order)
    order.sort(key=lambda v: degrees[v] + random.random() * 4.0)
    seq = np.zeros(n, dtype=np.int32)
    blocked = np.zeros(n, dtype=np.int32)
    for vertex in order:
        if blocked[vertex] == 0:
            seq[vertex] = 1
            for neighbor in adj[vertex]:
                blocked[neighbor] = 1
    return seq


def run(env_data, state, calc_makespan_fn):
    """Keep the largest of six inexpensive randomised greedy starts."""
    n = env_data["num_vertices"]
    current = np.asarray(state.sequence).ravel()
    if len(current) == n and np.isfinite(float(state.makespan)):
        return state
    adj = env_data["adj"]
    degrees = np.asarray([len(adj[v]) for v in range(n)], dtype=np.float64)
    starts = [_construct(n, adj, degrees) for _ in range(6)]
    seq = max(starts, key=lambda candidate: int(candidate.sum()))
    score = calc_makespan_fn(seq, env_data)
    state.sequence = seq.copy()
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
