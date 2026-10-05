import random


def run(env_data, state, calc_makespan_fn):
    num_nodes = env_data["num_nodes"]
    seq = list(state.sequence)
    # Flip a small fraction of random nodes
    mutation_rate = 0.02
    changed = False
    for i in range(num_nodes):
        if random.random() < mutation_rate:
            seq[i] = 1 - seq[i]
            changed = True

    if not changed and num_nodes > 0:
        i = random.randrange(num_nodes)
        seq[i] = 1 - seq[i]
        changed = True

    if changed:
        state.sequence = seq
        state.makespan = calc_makespan_fn(seq, env_data)
    return state
