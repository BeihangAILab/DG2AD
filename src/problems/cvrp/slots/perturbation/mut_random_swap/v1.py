import random


def run(env_data, state, calc_makespan_fn):
    seq = state.sequence[:]
    cust_indices = [i for i, x in enumerate(seq) if x != 0]
    if len(cust_indices) >= 2:
        i, j = random.sample(cust_indices, 2)
        seq[i], seq[j] = seq[j], seq[i]

    ms = calc_makespan_fn(seq, env_data)
    if ms < 999999999:
        state.sequence = seq
        state.makespan = ms
    return state
