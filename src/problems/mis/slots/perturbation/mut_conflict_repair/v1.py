import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Insert an outside vertex, remove its conflicts, then legally repair."""
    n = env_data["num_vertices"]
    adj = env_data["adj"]
    original = np.asarray(state.sequence, dtype=np.int32).copy()
    selected = set(np.flatnonzero(original).tolist())
    state.metadata["prev_sequence"] = original.copy()
    state.metadata["prev_makespan"] = state.makespan

    outside = [v for v in range(n) if v not in selected]
    if outside:
        incoming = random.choice(outside)
        chosen = (selected - (adj[incoming] & selected)) | {incoming}
        candidates = [v for v in range(n) if v not in chosen]
        random.shuffle(candidates)
        for vertex in candidates[: max(8, n // 5)]:
            if not (adj[vertex] & chosen):
                chosen.add(vertex)
    elif selected:
        chosen = selected - {random.choice(tuple(selected))}
    else:
        chosen = set()

    candidate = np.zeros(n, dtype=np.int32)
    if chosen:
        candidate[list(chosen)] = 1
    state.sequence = candidate
    state.makespan = calc_makespan_fn(candidate, env_data)
    return state
