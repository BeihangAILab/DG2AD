import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Combine current and elite sets, repairing conflicts during construction."""
    n = env_data["num_vertices"]
    adj = env_data["adj"]
    original = np.asarray(state.sequence, dtype=np.int32).copy()
    elite_raw = state.metadata.get("best_seq")
    elite = np.asarray(elite_raw, dtype=np.int32).ravel() if elite_raw is not None else original
    if len(elite) != n:
        elite = original
    state.metadata["prev_sequence"] = original.copy()
    state.metadata["prev_makespan"] = state.makespan

    current_set = set(np.flatnonzero(original).tolist())
    elite_set = set(np.flatnonzero(elite).tolist())
    chosen = current_set & elite_set
    priorities = list((current_set | elite_set) - chosen)
    random.shuffle(priorities)
    priorities.extend(v for v in range(n) if v not in current_set | elite_set)
    for vertex in priorities:
        if not (adj[vertex] & chosen):
            chosen.add(vertex)

    candidate = np.zeros(n, dtype=np.int32)
    if chosen:
        candidate[list(chosen)] = 1
    if np.array_equal(candidate, original) and n:
        selected = np.flatnonzero(candidate)
        if len(selected):
            candidate[random.choice(selected.tolist())] = 0
        else:
            candidate[random.randrange(n)] = 1
    state.sequence = candidate
    state.makespan = calc_makespan_fn(candidate, env_data)
    return state
