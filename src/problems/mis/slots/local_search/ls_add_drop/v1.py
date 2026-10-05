import random
import numpy as np


def _greedy_complete(selected, n, adj, excluded=None):
    chosen = set(selected)
    excluded = excluded or set()
    candidates = [v for v in range(n) if v not in chosen and v not in excluded]
    random.shuffle(candidates)
    candidates.sort(key=lambda v: len(adj[v]))
    for vertex in candidates:
        if not (adj[vertex] & chosen):
            chosen.add(vertex)
    return chosen


def run(env_data, state, calc_makespan_fn):
    """Add free vertices, then try bounded drop-and-greedy-complete moves."""
    n = env_data["num_vertices"]
    adj = env_data["adj"]
    original = np.asarray(state.sequence, dtype=np.int32)
    selected = set(np.flatnonzero(original).tolist())
    best_seq = original.copy()
    best_score = state.makespan

    trials = [_greedy_complete(selected, n, adj)]
    removals = list(selected)
    random.shuffle(removals)
    for removed in removals[:16]:
        trials.append(_greedy_complete(selected - {removed}, n, adj, {removed}))

    for candidate_set in trials:
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
