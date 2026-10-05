import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Move one item to a different priority position without greedy filtering."""
    n = env_data["num_activities"]
    if n < 2:
        return state
    original = np.asarray(state.sequence, dtype=np.int32).copy()
    state.metadata["prev_sequence"] = original.copy()
    state.metadata["prev_makespan"] = state.makespan
    source, target = random.sample(range(n), 2)
    values = original.tolist()
    item = values.pop(source)
    values.insert(target, item)
    candidate = np.asarray(values, dtype=np.int32)
    state.sequence = candidate
    state.makespan = calc_makespan_fn(candidate, env_data)
    return state
