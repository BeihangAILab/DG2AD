import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    n = env_data["num_vertices"]
    adj = env_data["adj"]
    seq = np.array(state.sequence, dtype=np.int32)
    best_seq = seq.copy()
    best_mk = state.makespan

    for _ in range(30):
        r = random.random()
        if r < 0.6:
            ones = [i for i in range(n) if seq[i] == 1]
            if not ones:
                continue
            v_remove = random.choice(ones)
            seq[v_remove] = 0
            blocked = np.zeros(n, dtype=np.int32)
            for i in range(n):
                if seq[i] == 1:
                    for u in adj[i]:
                        blocked[u] = 1
            degrees = np.array([len(adj[i]) for i in range(n)], dtype=np.int32)
            candidates = [i for i in range(n) if seq[i] == 0 and blocked[i] == 0]
            candidates.sort(key=lambda i: degrees[i])
            for c in candidates:
                if blocked[c] == 0:
                    seq[c] = 1
                    for u in adj[c]:
                        blocked[u] = 1
        else:
            blocked = np.zeros(n, dtype=np.int32)
            for i in range(n):
                if seq[i] == 1:
                    for u in adj[i]:
                        blocked[u] = 1
            candidates = [i for i in range(n) if seq[i] == 0 and blocked[i] == 0]
            if not candidates:
                continue
            degrees = np.array([len(adj[i]) for i in range(n)], dtype=np.int32)
            candidates.sort(key=lambda i: degrees[i])
            for c in candidates[:5]:
                seq[c] = 1
                mk = calc_makespan_fn(seq, env_data)
                if mk < best_mk:
                    best_mk = mk
                    best_seq = seq.copy()
                    break
                seq[c] = 0

        mk = calc_makespan_fn(seq, env_data)
        if mk < best_mk:
            best_mk, best_seq = mk, seq.copy()
        else:
            seq = best_seq.copy()

    state.sequence, state.makespan = best_seq, best_mk
    return state
