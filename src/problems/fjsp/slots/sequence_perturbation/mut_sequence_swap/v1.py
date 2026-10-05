import random
import domain_evaluator


def run(env_data, state, calc_makespan_fn):
    seq = list(state.sequence)
    alloc = state.metadata["machine_alloc"].copy()

    if len(seq) < 2:
        return state

    # Choose a random index i, then pick a random index j containing a different job
    i = random.randrange(len(seq))
    different_indices = [j for j in range(len(seq)) if seq[j] != seq[i]]
    if different_indices:
        j = random.choice(different_indices)
        seq[i], seq[j] = seq[j], seq[i]

    state.sequence = seq
    state.metadata["machine_alloc"] = alloc

    domain_evaluator.set_active_machine_allocation(alloc)
    state.makespan = calc_makespan_fn(seq, env_data)
    return state
