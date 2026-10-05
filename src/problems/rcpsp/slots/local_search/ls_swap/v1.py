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


def run(env_data, state, calc_makespan_fn):
    """
    Topology-preserving 2-opt local search for RCPSP.

    Tries all valid pairwise swaps and accepts the first improvement
    found. Repeats until a local optimum is reached (first-improvement).
    """
    n = env_data["num_activities"]
    adj = env_data["adj_matrix"]
    seq = np.array(state.sequence, dtype=np.int32).copy()
    best_mk = state.makespan

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, min(n, i + 50)):
                if not _is_valid_swap(seq, adj, i, j):
                    continue
                seq[i], seq[j] = seq[j], seq[i]
                mk = calc_makespan_fn(seq, env_data)
                if mk < best_mk:
                    best_mk = mk
                    improved = True
                    break
                seq[i], seq[j] = seq[j], seq[i]  # revert
            if improved:
                break

    state.sequence = seq
    state.makespan = best_mk
    return state
