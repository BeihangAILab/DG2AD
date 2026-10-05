import numpy as np
import domain_evaluator


def run(env_data, state, calc_makespan_fn):
    seq = list(state.sequence)

    meta = state.metadata
    g_best_alloc = meta.get("g_best_alloc")

    if g_best_alloc is None:
        return state

    # Defensive conversion to 2D numpy array
    alloc = np.ascontiguousarray(np.array(g_best_alloc, dtype=np.int32))

    state.sequence = seq
    state.metadata["machine_alloc"] = alloc

    domain_evaluator.set_active_machine_allocation(alloc)
    state.makespan = calc_makespan_fn(seq, env_data)
    return state
