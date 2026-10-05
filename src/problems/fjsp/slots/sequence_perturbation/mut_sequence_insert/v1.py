import random
import domain_evaluator


def run(env_data, state, calc_makespan_fn):
    seq = list(state.sequence)
    alloc = state.metadata["machine_alloc"].copy()
    n = len(seq)

    if n < 2:
        return state

    # Randomly extract one operation
    remove_idx = random.randrange(n)
    element = seq.pop(remove_idx)

    # Insert it into the front 20% of the remaining sequence
    max_insert_idx = max(1, int(len(seq) * 0.2))
    insert_idx = random.randint(0, min(len(seq), max_insert_idx))
    seq.insert(insert_idx, element)

    state.sequence = seq
    state.metadata["machine_alloc"] = alloc

    domain_evaluator.set_active_machine_allocation(alloc)
    state.makespan = calc_makespan_fn(seq, env_data)
    return state
