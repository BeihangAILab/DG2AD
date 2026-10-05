import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Create a uniformly random complete item priority permutation."""
    n = env_data["num_activities"]
    current = np.asarray(state.sequence).ravel()
    if len(current) == n and np.isfinite(float(state.makespan)):
        return state
    seq = np.random.permutation(n).astype(np.int32)
    score = calc_makespan_fn(seq, env_data)
    state.sequence = seq
    state.makespan = score
    state.metadata = {
        "temp": 5.0,
        "best_score": score,
        "best_seq": seq.copy(),
        "prev_makespan": score,
        "prev_sequence": seq.copy(),
        "no_improve": 0,
    }
    return state
