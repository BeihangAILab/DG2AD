import random


def run(env_data, state, calc_makespan_fn):
    demands = env_data["demands"]
    capacity = env_data["capacity"]
    num_customers = env_data["num_customers"]

    custs = list(range(1, num_customers + 1))
    random.shuffle(custs)

    seq = []
    curr_load = 0
    for c in custs:
        if curr_load + demands[c] > capacity:
            seq.append(0)
            curr_load = 0
        seq.append(c)
        curr_load += demands[c]

    state.sequence = seq
    state.makespan = calc_makespan_fn(seq, env_data)
    if not hasattr(state, "metadata") or state.metadata is None:
        state.metadata = {}
    return state
