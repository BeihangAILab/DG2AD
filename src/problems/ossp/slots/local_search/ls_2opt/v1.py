import numpy as np


def run(env_data, state, calc_makespan_fn):
    """
    2-opt local search for OSSP sequence.

    Tries all pairwise swaps and reverse-segment moves, accepting
    the first improvement found. Repeats until a local optimum
    is reached (first-improvement hill climbing).
    """
    n = env_data["total_ops"]
    seq = np.array(state.sequence, dtype=np.int32).copy()
    best_mk = state.makespan

    improved = True
    while improved:
        improved = False

        # Pairwise swaps
        for i in range(n - 1):
            for j in range(i + 1, n):
                seq[i], seq[j] = seq[j], seq[i]
                mk = calc_makespan_fn(seq, env_data)
                if mk < best_mk:
                    best_mk = mk
                    improved = True
                    break
                seq[i], seq[j] = seq[j], seq[i]  # revert
            if improved:
                break

    # Reverse sub-sequences
    if not improved:
        for i in range(n - 2):
            for length in range(2, min(7, n - i + 1)):
                j = i + length
                seq[i:j] = seq[i:j][::-1]
                mk = calc_makespan_fn(seq, env_data)
                if mk < best_mk:
                    best_mk = mk
                    improved = True
                    break
                seq[i:j] = seq[i:j][::-1]  # revert
            if improved:
                break

    state.sequence = seq
    state.makespan = best_mk
    return state
