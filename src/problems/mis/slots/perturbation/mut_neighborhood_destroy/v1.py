import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Destroy a selected neighborhood and refill without one anchor."""
    n = env_data["num_vertices"]
    adj = env_data["adj"]
    original = np.asarray(state.sequence, dtype=np.int32).copy()
    selected = set(np.flatnonzero(original).tolist())
    state.metadata["prev_sequence"] = original.copy()
    state.metadata["prev_makespan"] = state.makespan

    if selected:
        anchor = random.choice(tuple(selected))
        extra_count = min(len(selected) - 1, max(0, len(selected) // 8))
        extras = set(random.sample(list(selected - {anchor}), extra_count))
        chosen = selected - extras - {anchor}
        candidates = [v for v in range(n) if v not in chosen and v != anchor]
        random.shuffle(candidates)
        candidates.sort(key=lambda v: len(adj[v]))
        for vertex in candidates:
            if not (adj[vertex] & chosen):
                chosen.add(vertex)
    elif n:
        chosen = {random.randrange(n)}
    else:
        chosen = set()

    candidate = np.zeros(n, dtype=np.int32)
    if chosen:
        candidate[list(chosen)] = 1
    state.sequence = candidate
    state.makespan = calc_makespan_fn(candidate, env_data)
    return state
