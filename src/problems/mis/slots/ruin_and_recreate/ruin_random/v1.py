import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Ruin 25-45 percent of the set and recreate a legal maximal set."""
    n = env_data["num_vertices"]
    adj = env_data["adj"]
    original = np.asarray(state.sequence, dtype=np.int32).copy()
    selected = set(np.flatnonzero(original).tolist())
    state.metadata["prev_sequence"] = original.copy()
    state.metadata["prev_makespan"] = state.makespan

    if selected:
        count = max(1, int(round(len(selected) * random.uniform(0.25, 0.45))))
        removed = set(random.sample(list(selected), min(count, len(selected))))
        forbidden = {random.choice(tuple(removed))}
        chosen = selected - removed
    else:
        chosen = set()
        forbidden = set()
    order = [v for v in range(n) if v not in chosen and v not in forbidden]
    random.shuffle(order)
    order.sort(key=lambda v: len(adj[v]) + random.random() * 3.0)
    for vertex in order:
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
