import random


def run(env_data, state, calc_makespan_fn):
    seq, val = list(state.sequence), state.makespan
    b_seq, b_val = seq[:], val
    tabu, tenure = set(), 15
    for _ in range(100):
        moves = random.sample(range(len(seq) - 1), min(20, len(seq) - 1))
        best_n_v, best_n_s, best_n_k = float("inf"), None, None
        for i in moves:
            if seq[i] == seq[i + 1]:
                continue
            key = (min(seq[i], seq[i + 1]), i)
            if key in tabu:
                continue
            ns = seq[:]
            ns[i], ns[i + 1] = ns[i + 1], ns[i]
            v = calc_makespan_fn(ns, env_data)
            if v < best_n_v:
                best_n_v, best_n_s, best_n_k = v, ns, key
        if not best_n_s:
            break
        seq, val = best_n_s, best_n_v
        tabu.add(best_n_k)
        if len(tabu) > tenure:
            tabu.discard(next(iter(tabu)))
        if val < b_val:
            b_val, b_seq = val, seq[:]
    state.sequence, state.makespan = b_seq, b_val
    return state
