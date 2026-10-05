import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """
    Initialize a random TSP tour.

    Generates a random permutation of all cities. Multiple trials
    are performed and the best (shortest) tour is kept.
    """
    n = env_data["num_nodes"]

    best_seq = None
    best_cost = float("inf")

    for _ in range(20):
        seq = list(range(n))
        random.shuffle(seq)
        arr = np.array(seq, dtype=np.int32)
        cost = calc_makespan_fn(arr, env_data)
        if cost < best_cost:
            best_cost = cost
            best_seq = arr

    state.sequence = best_seq
    state.makespan = best_cost
    return state
