import numpy as np


def run(env_data, state, calc_makespan_fn):
    """
    Initialize a priority sequence sorted by item volume descending.

    Items with larger volume are packed first, which is a common
    heuristic for container loading problems -- large items are
    harder to fit later and block more space.
    """
    n = env_data["num_activities"]
    current = np.asarray(state.sequence).ravel()
    if len(current) == n and np.isfinite(float(state.makespan)):
        return state
    volumes = env_data["item_volumes"]
    # Sort items by volume descending (largest first)
    seq = np.argsort(-volumes).astype(np.int32)
    state.sequence = seq
    state.makespan = calc_makespan_fn(state, env_data)
    state.metadata = {
        "temp": 5.0,
        "best_score": state.makespan,
        "best_seq": seq.copy(),
        "prev_makespan": state.makespan,
        "prev_sequence": seq.copy(),
        "no_improve": 0,
    }
    return state
