import math
import random


def run(env_data, state, calc_makespan_fn):
    metadata = state.metadata
    current_ms = float(state.makespan)
    previous_seq = list(metadata.get("prev_sequence", state.sequence))
    previous_ms = float(metadata.get("prev_makespan", current_ms))

    temperature = float(metadata.get("sa_temp", 100.0))
    accept = current_ms <= previous_ms
    if not accept:
        probability = math.exp(-(current_ms - previous_ms) / max(temperature, 1e-9))
        accept = random.random() < probability

    if not accept:
        state.sequence = previous_seq
        state.makespan = previous_ms

    if "sa_best_ms" not in metadata:
        if previous_ms <= state.makespan:
            metadata["sa_best_ms"] = previous_ms
            metadata["sa_best_seq"] = previous_seq
        else:
            metadata["sa_best_ms"] = float(state.makespan)
            metadata["sa_best_seq"] = list(state.sequence)

    best_ms = float(metadata["sa_best_ms"])
    if state.makespan < best_ms:
        metadata["sa_best_ms"] = float(state.makespan)
        metadata["sa_best_seq"] = list(state.sequence)

    metadata["sa_temp"] = max(0.01, temperature * 0.995)
    return state
