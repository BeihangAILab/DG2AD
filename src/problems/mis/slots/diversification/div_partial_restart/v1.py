import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Keep a random core of the set and rebuild around a forbidden anchor."""
    n = env_data["num_vertices"]
    adj = env_data["adj"]
    original = np.asarray(state.sequence, dtype=np.int32).copy()
    selected = set(np.flatnonzero(original).tolist())
    state.metadata["prev_sequence"] = original.copy()
    state.metadata["prev_makespan"] = state.makespan

    if selected:
        keep_count = max(0, int(len(selected) * 0.6))
        chosen = set(random.sample(list(selected), keep_count))
        removed = selected - chosen
        forbidden = {random.choice(tuple(removed))} if removed else set()
    else:
        chosen = set()
        forbidden = set()
    candidates = [v for v in range(n) if v not in chosen and v not in forbidden]
    random.shuffle(candidates)
    for vertex in candidates:
        if not (adj[vertex] & chosen):
            chosen.add(vertex)

    if not selected and n and not chosen:
        chosen.add(random.randrange(n))
    candidate = np.zeros(n, dtype=np.int32)
    if chosen:
        candidate[list(chosen)] = 1
    state.sequence = candidate
    state.makespan = calc_makespan_fn(candidate, env_data)
    return state
