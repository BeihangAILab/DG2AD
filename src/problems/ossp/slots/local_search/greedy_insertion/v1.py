import numpy as np


def run(env_data, state, calc_makespan_fn):
    N = env_data["num_jobs"]
    M = env_data["num_machines"]
    n_ops = N * M
    env_data["ossp_times"]

    best_seq = state.sequence.copy()
    best_mk = float(state.makespan)

    md = state.metadata
    if "temperature" not in md:
        md["temperature"] = best_mk * 0.25
        md["best_global"] = best_mk
        md["best_global_seq"] = best_seq.copy()
        md["call_count"] = 0

    md["call_count"] = md.get("call_count", 0) + 1
    temp = md["temperature"]
    global_best_mk = md["best_global"]
    global_best_seq = md.get("best_global_seq", best_seq.copy())

    if best_mk < global_best_mk:
        global_best_mk = best_mk
        global_best_seq = best_seq.copy()

    current_seq = best_seq.copy()
    current_mk = best_mk

    iterations = 50
    for it in range(iterations):
        r = np.random.random()
        new_seq = current_seq.copy()

        if r < 0.35:
            # Swap two random positions
            i, j = np.random.randint(0, n_ops), np.random.randint(0, n_ops - 1)
            if j >= i:
                j += 1
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        elif r < 0.6:
            # Insert move: remove element at i, insert at j
            i = np.random.randint(0, n_ops)
            val = new_seq[i]
            new_seq = np.delete(new_seq, i)
            j = np.random.randint(0, n_ops)
            new_seq = np.insert(new_seq, j, val)
        elif r < 0.8:
            # Or-opt: remove segment of 2-3, reinsert
            seg_len = np.random.randint(2, min(4, n_ops))
            i = np.random.randint(0, n_ops - seg_len)
            seg = new_seq[i : i + seg_len].copy()
            new_seq = np.delete(new_seq, range(i, i + seg_len))
            j = np.random.randint(0, len(new_seq) + 1)
            new_seq = np.insert(new_seq, j, seg)
        else:
            # Reverse a sub-segment (2-opt style)
            i = np.random.randint(0, n_ops - 1)
            length = np.random.randint(2, min(6, n_ops - i + 1))
            j = i + length
            new_seq[i:j] = new_seq[i:j][::-1]

        state.sequence = new_seq
        mk = calc_makespan_fn(state, env_data)

        delta = mk - current_mk
        if delta < 0:
            current_seq = new_seq
            current_mk = mk
            if mk < best_mk:
                best_mk = mk
                best_seq = new_seq.copy()
                if mk < global_best_mk:
                    global_best_mk = mk
                    global_best_seq = new_seq.copy()
        elif temp > 0.1 and np.random.random() < np.exp(-delta / temp):
            current_seq = new_seq
            current_mk = mk

    # Adaptive cooling with periodic reheat
    temp *= 0.88
    if md["call_count"] % 12 == 0:
        temp = global_best_mk * 0.2
    elif temp < 0.5:
        temp = global_best_mk * 0.08

    md["temperature"] = temp
    md["best_global"] = global_best_mk
    md["best_global_seq"] = global_best_seq.copy()

    if global_best_mk < best_mk:
        state.sequence = global_best_seq.copy()
        state.makespan = global_best_mk
    else:
        state.sequence = best_seq
        state.makespan = best_mk
    return state
