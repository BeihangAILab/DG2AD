import random


def run(env_data, state, calc_makespan_fn):
    num_nodes = env_data["num_nodes"]
    # Initialize a random cut assignment (0 or 1 for each node)
    state.sequence = [random.choice([0, 1]) for _ in range(num_nodes)]
    state.makespan = calc_makespan_fn(state.sequence, env_data)
    if not hasattr(state, "metadata") or state.metadata is None:
        state.metadata = {}
    return state
