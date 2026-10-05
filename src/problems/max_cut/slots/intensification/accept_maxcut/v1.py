def run(env_data, state, calc_makespan_fn):
    # Standard Hill Climbing: only accept better or equal solutions, otherwise roll back
    prev_makespan = state.metadata.get("prev_makespan", float("inf"))
    if state.makespan > prev_makespan:
        # Rollback
        state.sequence = state.metadata["prev_sequence"].copy()
        state.makespan = prev_makespan
    return state
