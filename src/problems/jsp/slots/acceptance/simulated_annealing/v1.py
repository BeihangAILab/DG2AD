import math
import random


def run(env_data, state, calc_makespan_fn):
    meta = state.metadata
    prev_ms, prev_seq = meta.get("prev_makespan", float("inf")), meta.get("prev_sequence")
    g_best_ms, g_best_seq = meta.get("g_best_ms", float("inf")), meta.get("g_best_seq")
    temp, no_imp = meta.get("temp", 30.0), meta.get("no_improve", 0)
    if state.makespan < g_best_ms:
        g_best_ms, g_best_seq, no_imp = state.makespan, state.sequence[:], 0
    else:
        no_imp += 1
    delta = state.makespan - prev_ms
    if delta < 0 or random.random() < math.exp(-delta / max(0.1, temp)):
        pass
    elif prev_seq is not None:
        state.sequence, state.makespan = prev_seq[:], prev_ms
    temp *= 0.999
    if no_imp > 25:
        if g_best_seq is not None:
            state.sequence, state.makespan = g_best_seq[:], g_best_ms
        temp, no_imp, meta["force_reset"] = 30.0, 0, True
    meta.update(
        {"temp": temp, "no_improve": no_imp, "g_best_ms": g_best_ms, "g_best_seq": g_best_seq}
    )
    return state
