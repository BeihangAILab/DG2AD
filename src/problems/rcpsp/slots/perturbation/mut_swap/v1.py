import random
import numpy as np


def _topological_swap(seq, adj, n, attempts=20):
    """
    Attempt a topology-preserving swap of two positions.

    Swaps seq[i] and seq[j] only if the result is still topologically valid.
    A swap is valid iff for all edges (a->b), a still appears before b.
    """
    for _ in range(attempts):
        i, j = random.sample(range(n), 2)
        if i > j:
            i, j = j, i
        # Check if swapping i and j would violate any edges BETWEEN them
        u, v = seq[i], seq[j]
        if adj[u, v]:
            continue  # u must precede v, cannot swap
        # Check edges from any node between i and j
        valid = True
        for k in range(i + 1, j):
            w = seq[k]
            if adj[w, v]:  # w must precede v, but after swap v moves earlier
                valid = False
                break
            if adj[u, w]:  # u must precede w, but after swap u moves later
                valid = False
                break
        if valid:
            seq[i], seq[j] = seq[j], seq[i]
            return True
    return False


def run(env_data, state, calc_makespan_fn):
    """
    Perturb the RCPSP sequence via topology-preserving random swaps.

    Saves the previous state for potential rollback by acceptance.
    """
    n = env_data["num_activities"]
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
