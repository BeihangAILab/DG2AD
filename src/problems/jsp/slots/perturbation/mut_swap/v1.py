import random


def run(env_data, state, calc_makespan_fn):
    seq = list(state.sequence)
    pairs = [(i, j) for i in range(len(seq)) for j in range(i + 1, len(seq)) if seq[i] != seq[j]]
    if not pairs:
        return state
    i, j = random.choice(pairs)
    seq[i], seq[j] = seq[j], seq[i]
    state.sequence, state.makespan = seq, calc_makespan_fn(seq, env_data)
    return state
