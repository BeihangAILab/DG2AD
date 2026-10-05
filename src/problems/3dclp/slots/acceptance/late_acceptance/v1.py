import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Late-acceptance hill climbing relative to the immediate parent solution."""
    del env_data, calc_makespan_fn
    metadata = state.metadata
    parent_score = float(metadata.get("prev_makespan", state.makespan))
    parent_sequence = metadata.get("prev_sequence")
    history = list(metadata.get("late_history", [parent_score] * 20))
    if len(history) != 20:
        history = [parent_score] * 20
    index = int(metadata.get("late_index", 0)) % len(history)
    current_score = float(state.makespan)

    accepted = current_score <= parent_score or current_score <= history[index]
    if not accepted and parent_sequence is not None:
        state.sequence = np.asarray(parent_sequence, dtype=np.int32).copy()
        state.makespan = parent_score

    best_score = float(metadata.get("best_score", state.makespan))
    best_sequence = metadata.get("best_seq")
    if state.makespan < best_score:
        best_score = float(state.makespan)
        best_sequence = np.asarray(state.sequence, dtype=np.int32).copy()

    history[index] = float(state.makespan)
    metadata["late_history"] = history
    metadata["late_index"] = (index + 1) % len(history)
    metadata["best_score"] = best_score
    metadata["best_seq"] = best_sequence
    metadata["prev_makespan"] = float(state.makespan)
    metadata["prev_sequence"] = np.asarray(state.sequence, dtype=np.int32).copy()
    return state
