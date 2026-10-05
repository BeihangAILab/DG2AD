import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """
    Initialize with a random topologically-sorted sequence.

    Uses Kahn's algorithm with random selection to produce diverse
    valid topological orderings. Multiple trials are performed and
    the best (lowest makespan) is kept.
    """
    n = env_data["num_activities"]
    adj = env_data["adj_matrix"]

    best_seq = None
    best_mk = float("inf")

    for _ in range(10):
        in_degree = np.sum(adj, axis=0)
        eligible = [i for i in range(n) if in_degree[i] == 0]
        seq = []

        while eligible:
            idx = random.randrange(len(eligible))
            act = eligible.pop(idx)
            seq.append(act)
            for j in range(n):
                if adj[act, j]:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        eligible.append(j)

        arr = np.array(seq, dtype=np.int32)
        mk = calc_makespan_fn(arr, env_data)
        if mk < best_mk:
            best_mk = mk
            best_seq = arr

    state.sequence = best_seq
    state.makespan = best_mk
    return state
