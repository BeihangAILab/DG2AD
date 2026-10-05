import numpy as np


def run(env_data, state, calc_makespan_fn):
    """
    2-opt local search for TSP — first improvement, limited evaluations.

    Uses efficient delta evaluation: only recomputes the delta between
    old and new edges rather than the full tour length. Capped at 500
    evaluations to avoid excessive runtime.
    """
    seq = list(state.sequence)
    n = env_data["num_nodes"]
    dist = env_data["distance_matrix"]
    cost = state.makespan

    improved = True
    evals = 0
    max_evals = 500

    while improved and evals < max_evals:
        improved = False
        for i in range(n - 2):
            a_val = seq[i]
            b_val = seq[i + 1]
            for j in range(i + 2, n):
                evals += 1
                if evals > max_evals:
                    break

                c_val = seq[j]
                d_val = seq[(j + 1) % n]

                # Delta: remove (a,b)+(c,d), add (a,c)+(b,d)
                gain = (dist[a_val][b_val] + dist[c_val][d_val]) - (
                    dist[a_val][c_val] + dist[b_val][d_val]
                )

                if gain > 1e-9:
                    seq[i + 1 : j + 1] = seq[i + 1 : j + 1][::-1]
                    cost -= gain
                    improved = True
                    break
            if improved or evals > max_evals:
                break

    state.sequence = np.array(seq, dtype=np.int32)
    state.makespan = cost
    return state
