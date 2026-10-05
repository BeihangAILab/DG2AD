import numpy as np


def run(env_data, state, calc_makespan_fn):
    """
    2-opt local search on the priority sequence.

    Tries all pairwise swaps and accepts the first improvement found.
    Repeats until no improving move exists (first-improvement hill climbing).
    """
    n = env_data["num_activities"]
    seq = np.array(state.sequence, dtype=np.int32).copy()
    best_mk = state.makespan
    best_seq = seq.copy()

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                seq[i], seq[j] = seq[j], seq[i]
                mk = calc_makespan_fn(seq, env_data)
                if mk < best_mk:
                    best_mk = mk
                    best_seq = seq.copy()
                    improved = True
                    break  # First-improvement, restart outer loop
                seq[i], seq[j] = seq[j], seq[i]  # Revert
            if improved:
                seq = best_seq.copy()
                break

    state.sequence = best_seq
    state.makespan = best_mk
    return state
