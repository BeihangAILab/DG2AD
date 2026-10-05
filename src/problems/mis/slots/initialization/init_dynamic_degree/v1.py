import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Randomised dynamic minimum-residual-degree construction."""
    n = env_data["num_vertices"]
    current = np.asarray(state.sequence).ravel()
    if len(current) == n and np.isfinite(float(state.makespan)):
        return state
    adj = env_data["adj"]
    candidates = set(range(n))
    seq = np.zeros(n, dtype=np.int32)

    while candidates:
        sample = random.sample(list(candidates), min(64, len(candidates)))
        residual = [(len(adj[v] & candidates), random.random(), v) for v in sample]
        vertex = min(residual)[2]
        seq[vertex] = 1
        candidates.discard(vertex)
        candidates.difference_update(adj[vertex])

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
