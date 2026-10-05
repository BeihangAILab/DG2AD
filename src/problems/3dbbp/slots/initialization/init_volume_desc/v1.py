import numpy as np


def run(env_data, state, calc_makespan_fn):
    """
    Initialize a priority sequence sorted by item volume descending.

    Items with larger volume are packed first, which is a common
    heuristic for bin packing problems.
    """
    volumes = env_data["item_volumes"]
    # Sort items by volume descending (largest first)
    seq = np.argsort(-volumes).astype(np.int32)
    state.sequence = seq
    state.makespan = calc_makespan_fn(state, env_data)
    return state
