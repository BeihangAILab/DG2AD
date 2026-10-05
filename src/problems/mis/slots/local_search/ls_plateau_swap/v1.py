import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Traverse a 1-for-1 plateau internally, committing only an improvement."""
    n = env_data["num_vertices"]
    adj = env_data["adj"]
    original = np.asarray(state.sequence, dtype=np.int32)
    selected = set(np.flatnonzero(original).tolist())
    best_seq = original.copy()
    best_score = state.makespan
    outside = [v for v in range(n) if v not in selected]
    random.shuffle(outside)

    for incoming in outside[:32]:
        conflicts = adj[incoming] & selected
        if len(conflicts) != 1:
            continue
        plateau = (selected - conflicts) | {incoming}
        additions = [v for v in range(n) if v not in plateau and not (adj[v] & plateau)]
        random.shuffle(additions)
        additions.sort(key=lambda v: len(adj[v]))
        for vertex in additions:
            if not (adj[vertex] & plateau):
                plateau.add(vertex)
        if len(plateau) <= len(selected):
            continue
        candidate = np.zeros(n, dtype=np.int32)
        candidate[list(plateau)] = 1
        score = calc_makespan_fn(candidate, env_data)
        if score < best_score:
            best_seq, best_score = candidate, score
            break

    state.sequence = best_seq
    state.makespan = best_score
    return state
