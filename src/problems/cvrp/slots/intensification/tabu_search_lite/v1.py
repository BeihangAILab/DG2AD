import random
from collections import deque


def run(env_data, state, calc_makespan_fn):
    if not hasattr(state, "metadata") or state.metadata is None:
        state.metadata = {}

    if "tabu_list" not in state.metadata:
        state.metadata["tabu_list"] = deque(maxlen=10)

    seq = state.sequence[:]
    cust_indices = [i for i, x in enumerate(seq) if x != 0]
    if len(cust_indices) < 2:
        return state

    best_move = None
    best_ms = float("inf")

    for _ in range(20):  # sample 20 neighbors
        i, j = random.sample(cust_indices, 2)
        if (seq[i], seq[j]) in state.metadata["tabu_list"]:
            continue

        new_seq = seq[:]
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        ms = calc_makespan_fn(new_seq, env_data)
        if ms < best_ms:
            best_ms = ms
            best_move = (new_seq, i, j)

    if best_move and best_ms < 999999999:
        state.sequence = best_move[0]
        state.makespan = best_ms
        c1, c2 = seq[best_move[1]], seq[best_move[2]]
        state.metadata["tabu_list"].append((c1, c2))
        state.metadata["tabu_list"].append((c2, c1))

    return state
