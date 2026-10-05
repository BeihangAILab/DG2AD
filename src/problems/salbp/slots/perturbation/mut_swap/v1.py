import random
import numpy as np


def _is_valid_swap(seq, adj, i, j):
    """Check if swapping seq[i] and seq[j] preserves topological order."""
    if i > j:
        i, j = j, i
    u, v = seq[i], seq[j]
    if adj[u, v]:
        return False
    for k in range(i + 1, j):
        w = seq[k]
        if adj[w, v] or adj[u, w]:
            return False
    return True


def _topological_swap(seq, adj, n, attempts=20):
    """Attempt a topology-preserving swap of two positions."""
    for _ in range(attempts):
        i, j = random.sample(range(n), 2)
        if _is_valid_swap(seq, adj, i, j):
            seq[i], seq[j] = seq[j], seq[i]
            return True
    return False


def run(env_data, state, calc_makespan_fn):
    """
    Perturb the SALBP sequence via topology-preserving swaps.

    Saves the previous state for potential rollback by acceptance.
    """
    n = env_data["num_tasks"]
    adj = env_data["adj_matrix"]

    md = state.metadata
    md["prev_sequence"] = np.array(state.sequence, dtype=np.int32).copy()
    md["prev_makespan"] = state.makespan

    seq = np.array(state.sequence, dtype=np.int32).copy()
    k = max(2, int(n * random.uniform(0.03, 0.15)))

    for _ in range(k):
        _topological_swap(seq, adj, n, attempts=30)

    state.sequence = seq
    state.makespan = calc_makespan_fn(state, env_data)
    return state
