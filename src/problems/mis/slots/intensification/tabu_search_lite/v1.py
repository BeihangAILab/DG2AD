import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Bounded tabu-assisted 1-for-1 moves followed by greedy completion."""
    n = env_data["num_vertices"]
    adj = env_data["adj"]
    original = np.asarray(state.sequence, dtype=np.int32)
    selected = set(np.flatnonzero(original).tolist())
    best_seq = original.copy()
    best_score = state.makespan
    tabu = list(state.metadata.get("mis_tabu", []))[-16:]
    outside = [v for v in range(n) if v not in selected and v not in tabu]
    random.shuffle(outside)

    for incoming in outside[:24]:
        conflicts = adj[incoming] & selected
        if len(conflicts) != 1:
            continue
        removed = next(iter(conflicts))
        chosen = (selected - {removed}) | {incoming}
        candidates = [v for v in range(n) if v not in chosen and v != removed and v not in tabu]
        random.shuffle(candidates)
        candidates.sort(key=lambda v: len(adj[v]))
        for vertex in candidates:
            if not (adj[vertex] & chosen):
                chosen.add(vertex)
        if len(chosen) <= len(selected):
            continue
        candidate = np.zeros(n, dtype=np.int32)
        candidate[list(chosen)] = 1
        score = calc_makespan_fn(candidate, env_data)
        if score < best_score:
            best_seq, best_score = candidate, score
            tabu.append(removed)
            break

    state.metadata["mis_tabu"] = tabu[-16:]
    state.sequence = best_seq
    state.makespan = best_score
    return state
