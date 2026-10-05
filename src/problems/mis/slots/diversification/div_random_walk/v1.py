import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Perform a short feasible walk while keeping one removed anchor tabu."""
    n = env_data["num_vertices"]
    adj = env_data["adj"]
    original = np.asarray(state.sequence, dtype=np.int32).copy()
    chosen = set(np.flatnonzero(original).tolist())
    state.metadata["prev_sequence"] = original.copy()
    state.metadata["prev_makespan"] = state.makespan

    forbidden = set()
    if chosen:
        anchor = random.choice(tuple(chosen))
        chosen.remove(anchor)
        forbidden.add(anchor)
    elif n:
        chosen.add(random.randrange(n))

    for _ in range(max(2, min(8, n // 20 + 2))):
        feasible = [
            v for v in range(n) if v not in chosen and v not in forbidden and not (adj[v] & chosen)
        ]
        if feasible and (not chosen or random.random() < 0.7):
            chosen.add(random.choice(feasible))
        elif chosen:
            chosen.remove(random.choice(tuple(chosen)))

    candidate = np.zeros(n, dtype=np.int32)
    if chosen:
        candidate[list(chosen)] = 1
    state.sequence = candidate
    state.makespan = calc_makespan_fn(candidate, env_data)
    return state
