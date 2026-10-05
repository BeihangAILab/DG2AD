import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Replace one selected vertex and greedily add at least two vertices."""
    n = env_data["num_vertices"]
    adj = env_data["adj"]
    original = np.asarray(state.sequence, dtype=np.int32)
    selected = set(np.flatnonzero(original).tolist())
    best_seq = original.copy()
    best_score = state.makespan
    outside = [v for v in range(n) if v not in selected]
    random.shuffle(outside)

    for vertex in outside[:32]:
        conflicts = adj[vertex] & selected
        if len(conflicts) != 1:
            continue
        candidate_set = (selected - conflicts) | {vertex}
        additions = [
            v
            for v in outside
            if v != vertex and v not in candidate_set and not (adj[v] & candidate_set)
        ]
        additions.sort(key=lambda v: len(adj[v]))
        for extra in additions:
            if not (adj[extra] & candidate_set):
                candidate_set.add(extra)
        if len(candidate_set) <= len(selected):
            continue
        candidate = np.zeros(n, dtype=np.int32)
        candidate[list(candidate_set)] = 1
        score = calc_makespan_fn(candidate, env_data)
        if score < best_score:
            best_seq, best_score = candidate, score
            break

    state.sequence = best_seq
    state.makespan = best_score
    return state
