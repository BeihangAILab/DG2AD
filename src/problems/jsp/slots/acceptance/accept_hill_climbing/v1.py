def run(env_data, state, calc_makespan_fn):
    prev_ms = state.metadata.get("prev_makespan", float("inf"))
    prev_seq = state.metadata.get("prev_sequence", None)
    if state.makespan < prev_ms:
        pass
    elif prev_seq is not None:
        state.sequence, state.makespan = prev_seq[:], prev_ms
    return state
