import numpy as np


def run(env_data, state, calc_makespan_fn):
    n_jobs = env_data["num_jobs"]
    n_machines = env_data["num_machines"]
    total_ops = n_jobs * n_machines
    times = env_data["ossp_times"]
    best_seq = None
    best_mk = float("inf")

    def evaluate(seq):
        nonlocal best_seq, best_mk
        state.sequence = seq if isinstance(seq, np.ndarray) else np.array(seq, dtype=np.int32)
        mk = calc_makespan_fn(state, env_data)
        state.makespan = mk
        if mk < best_mk:
            best_mk = mk
            best_seq = state.sequence.copy()
        return mk

    def local_search(seq, budget):
        cur = seq.copy()
        cur_mk = evaluate(cur.copy())
        for _ in range(budget):
            i = np.random.randint(0, total_ops)
            j = np.random.randint(0, total_ops - 1)
            if j >= i:
                j += 1
            tmp = cur.copy()
            op = tmp[i]
            tmp = np.delete(tmp, i)
            tmp = np.insert(tmp, j if j < len(tmp) else len(tmp) - 1, op)
            new_mk = evaluate(tmp.astype(np.int32))
            if new_mk <= cur_mk:
                cur = tmp.astype(np.int32)
                cur_mk = new_mk
        return cur, cur_mk

    ops = np.arange(total_ops, dtype=np.int32)
    proc_times = np.array([times[op // n_machines, op % n_machines] for op in ops])
    evaluate(ops[np.argsort(proc_times)].astype(np.int32).copy())
    evaluate(ops[np.argsort(-proc_times)].astype(np.int32).copy())

    # Greedy ECT construction
    for trial in range(3):
        job_avail = np.zeros(n_jobs)
        mach_avail = np.zeros(n_machines)
        scheduled = np.zeros(total_ops, dtype=bool)
        seq_list = []
        for _ in range(total_ops):
            best_op, best_ct = -1, float("inf")
            candidates = np.where(~scheduled)[0]
            if trial > 0:
                np.random.shuffle(candidates)
            for op in candidates[: max(5, len(candidates) // (1 + trial))]:
                j, m = op // n_machines, op % n_machines
                ct = max(job_avail[j], mach_avail[m]) + times[j, m]
                if ct < best_ct or (ct == best_ct and np.random.random() < 0.3):
                    best_ct, best_op = ct, op
            j, m = best_op // n_machines, best_op % n_machines
            job_avail[j] = best_ct
            mach_avail[m] = best_ct
            scheduled[best_op] = True
            seq_list.append(best_op)
        evaluate(np.array(seq_list, dtype=np.int32))

    # Job-by-job SPT and interleaved
    sl = []
    for j in range(n_jobs):
        jo = [(times[j, m], j * n_machines + m) for m in range(n_machines)]
        jo.sort()
        sl.extend([o for _, o in jo])
    evaluate(np.array(sl, dtype=np.int32))
    job_order = np.argsort(times.sum(axis=1))
    sr = []
    for mi in np.argsort(times.sum(axis=0)):
        for j in job_order:
            sr.append(j * n_machines + mi)
    evaluate(np.array(sr, dtype=np.int32))

    # Multi-start random with insertion-based local search
    K = max(10, min(80, 200 - total_ops))
    for _ in range(K):
        seq = np.arange(total_ops, dtype=np.int32)
        np.random.shuffle(seq)
        local_search(seq, min(total_ops, 25))

    # Deep local search on best
    local_search(best_seq.copy(), min(total_ops * 4, 200))

    state.sequence = best_seq.copy()
    state.makespan = best_mk
    state.metadata["temp"] = max(best_mk * 0.04, 8.0)
    state.metadata["no_improve"] = 0
    state.metadata["best_makespan"] = best_mk
    state.metadata["best_sequence"] = best_seq.copy()
    state.metadata["iteration"] = 0
    return state
