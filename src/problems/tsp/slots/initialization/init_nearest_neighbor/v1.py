import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """
    Initialize a TSP tour using Nearest Neighbour heuristic.

    Tries multiple random starting cities (up to 5) and keeps the
    shortest tour found.
    """
    n = env_data["num_nodes"]
    dist = env_data["distance_matrix"]

    num_starts = min(5, n)
    start_cities = random.sample(range(n), num_starts)

    best_seq = None
    best_cost = float("inf")

    for start in start_cities:
        visited = [False] * n
        seq = [start]
        visited[start] = True
        current = start

        for _ in range(n - 1):
            row = dist[current]
            best_next = -1
            best_d = float("inf")
            for j in range(n):
                if not visited[j] and row[j] < best_d:
                    best_d = row[j]
                    best_next = j
            if best_next == -1:
                break
            seq.append(best_next)
            visited[best_next] = True
            current = best_next

        cost = calc_makespan_fn(np.array(seq, dtype=np.int32), env_data)
        if cost < best_cost:
            best_cost = cost
            best_seq = seq

    if best_seq is None:
        best_seq = list(range(n))
        best_cost = calc_makespan_fn(np.array(best_seq, dtype=np.int32), env_data)

    state.sequence = np.array(best_seq, dtype=np.int32)
    state.makespan = best_cost
    return state
