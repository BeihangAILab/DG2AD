import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Remove part of the set and greedily rebuild while forcing a change."""
    n = env_data["num_vertices"]
    adj = env_data["adj"]
    original = np.asarray(state.sequence, dtype=np.int32).copy()
    selected = set(np.flatnonzero(original).tolist())
    state.metadata["prev_sequence"] = original.copy()
    state.metadata["prev_makespan"] = state.makespan

    if not selected:
        rebuilt = {random.randrange(n)} if n else set()
    else:
        remove_count = max(1, int(round(len(selected) * random.uniform(0.1, 0.3))))
        removed = set(random.sample(list(selected), min(remove_count, len(selected))))
        forbidden = {random.choice(tuple(removed))}
        rebuilt = selected - removed
        candidates = [v for v in range(n) if v not in rebuilt and v not in forbidden]
        random.shuffle(candidates)
        candidates.sort(key=lambda v: len(adj[v]))
        for vertex in candidates:
            if not (adj[vertex] & rebuilt):
                rebuilt.add(vertex)

    candidate = np.zeros(n, dtype=np.int32)
    if rebuilt:
        candidate[list(rebuilt)] = 1
    state.sequence = candidate
    state.makespan = calc_makespan_fn(candidate, env_data)
    return state
