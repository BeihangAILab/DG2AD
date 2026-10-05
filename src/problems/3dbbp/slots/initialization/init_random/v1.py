import numpy as np
import random


def run(env_data, state, calc_makespan_fn):
    """
    Initialize a random priority sequence.

    All items are randomly shuffled to form the packing order.
    """
    n = env_data["num_activities"]
    seq = list(range(n))
    random.shuffle(seq)
    state.sequence = np.array(seq, dtype=np.int32)
    state.makespan = calc_makespan_fn(state, env_data)
    return state
