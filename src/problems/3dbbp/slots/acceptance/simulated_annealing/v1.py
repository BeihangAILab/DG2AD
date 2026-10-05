import math
import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """
    Simulated annealing acceptance criterion with adaptive cooling.

    Tracks best solution found and supports reheating on stagnation.
    The state.sequence and state.makespan reflect the CURRENT solution
    after the accept/reject decision.
    """
    md = state.metadata
    temp = md.setdefault("temp", 5.0)
    prev_mk = md.get("prev_makespan", state.makespan)
    best_mk = md.get("best_score", state.makespan)
    best_seq = md.get("best_seq")
    stagnation = md.get("no_improve", 0)
    accept_count = md.get("accept_count", 0)
    total_count = md.get("total_count", 0)

    cur_mk = state.makespan

    # Accept/reject decision
    if cur_mk < best_mk:
        best_mk = cur_mk
        best_seq = np.array(state.sequence, dtype=np.int32).copy()
        stagnation = 0
        accepted = True
    elif cur_mk <= prev_mk:
        stagnation += 1
        accepted = True
    else:
        delta = cur_mk - prev_mk
        if temp > 0.001 and random.random() < math.exp(-delta / temp):
            stagnation += 1
            accepted = True
        else:
            # Reject: rollback
            prev_seq = md.get("prev_sequence")
            if prev_seq is not None:
                state.sequence = np.asarray(prev_seq, dtype=np.int32)
            state.makespan = prev_mk
            stagnation += 1
            accepted = False

    total_count += 1
    if accepted:
        accept_count += 1

    # Adaptive cooling based on acceptance ratio
    if total_count % 10 == 0:
        ratio = accept_count / max(total_count, 1)
        if ratio > 0.6:
            cool = 0.95
        elif ratio > 0.4:
            cool = 0.97
        elif ratio > 0.2:
            cool = 0.99
        else:
            cool = 0.995
        md["cool"] = cool
    cool = md.get("cool", 0.97)
    temp *= cool

    # Reheating on prolonged stagnation
    if stagnation > 80:
        temp = max(temp, 3.0)
        if best_seq is not None:
            state.sequence = best_seq.copy()
        state.makespan = best_mk
        prev_mk = best_mk
        stagnation = 0
        accept_count = 0
        total_count = 0

    if temp < 0.001:
        temp = 0.001

    # Persist metadata
    md["temp"] = temp
    md["prev_makespan"] = state.makespan
    md["best_score"] = best_mk
    md["best_seq"] = best_seq
    md["no_improve"] = stagnation
    md["accept_count"] = accept_count
    md["total_count"] = total_count
    md["prev_sequence"] = np.array(state.sequence, dtype=np.int32).copy()
    return state
