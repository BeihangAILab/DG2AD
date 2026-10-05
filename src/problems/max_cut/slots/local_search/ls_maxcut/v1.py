import numpy as np
from src.problems.max_cut.domain_evaluator import get_flip_delta


def run(env_data, state, calc_makespan_fn):
    # One-pass greedy local search: flip any node that increases cut weight (decreases makespan)
    num_nodes = env_data["num_nodes"]
    neighbors = env_data["neighbors"]
    weights = env_data["edge_weights"]
    offsets = env_data["offsets"]

    # Convert sequence to numpy array for Numba JIT speed
    nodes = np.array(state.sequence, dtype=np.int32)
    makespan = state.makespan

    for i in range(num_nodes):
        delta = get_flip_delta(i, nodes, neighbors, weights, offsets)
        if delta > 0:  # delta > 0 means positive cut weight increases
            nodes[i] = 1 - nodes[i]
            makespan -= delta

    state.sequence = list(nodes)
    state.makespan = calc_makespan_fn(state.sequence, env_data)
    return state
