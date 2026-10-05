import random
import numpy as np
import domain_evaluator


def run(env_data, state, calc_makespan_fn):
    seq = list(state.sequence)
    # Defensive conversion to 2D numpy array
    alloc = np.ascontiguousarray(np.array(state.metadata["machine_alloc"], dtype=np.int32))
    n = len(seq)

    if n < 2:
        return state

    domain_evaluator.set_active_machine_allocation(alloc)

    best_seq = seq[:]
    best_ms = state.makespan

    # Find all indices where adjacent elements are different
    valid_indices = [i for i in range(n - 1) if seq[i] != seq[i + 1]]
    if not valid_indices:
        return state

    # Try up to 20 unique adjacent swaps
    random.shuffle(valid_indices)
    for i in valid_indices[:20]:
        trial_seq = seq[:]
        trial_seq[i], trial_seq[i + 1] = trial_seq[i + 1], trial_seq[i]

        ms = calc_makespan_fn(trial_seq, env_data)
        if ms < best_ms:
            best_ms = ms
            best_seq = trial_seq[:]
            break  # First-improvement strategy

    state.sequence = best_seq
    state.metadata["machine_alloc"] = alloc

    state.makespan = best_ms
    return state
