import random
import numpy as np


def _extract_critical_path(seq, env_data):
    """Trace the critical path of a JSP schedule. Returns list of (job, op) tuples."""
    nj = env_data["num_jobs"]
    nm = env_data["num_machines"]
    machines = env_data["machines_matrix"]
    times = env_data["times_matrix"]

    n = len(seq)
    seq_arr = np.asarray(seq, dtype=np.int32)
    job_op = np.zeros(nj, dtype=np.int32)
    job_end = np.zeros(nj, dtype=np.float64)
    mach_end = np.zeros(nm, dtype=np.float64)

    sched_m = np.zeros(n, dtype=np.int32)
    sched_s = np.zeros(n, dtype=np.float64)
    sched_e = np.zeros(n, dtype=np.float64)
    pred_j = np.zeros(n, dtype=np.int8)
    pred_m = np.zeros(n, dtype=np.int8)
    job_to_idx = np.full((nj, nm), -1, dtype=np.int32)

    for pos in range(n):
        job = int(seq_arr[pos])
        op = int(job_op[job])
        if op >= nm:
            return []
        m = int(machines[job, op])
        t = float(times[job, op])
        je = job_end[job]
        me = mach_end[m]
        start = je if je > me else me
        end = start + t
        sched_m[pos] = m
        sched_s[pos] = start
        sched_e[pos] = end
        pred_j[pos] = abs(je - start) < 1e-9
        pred_m[pos] = abs(me - start) < 1e-9
        job_end[job] = end
        mach_end[m] = end
        job_to_idx[job, op] = pos
        job_op[job] += 1

    makespan = float(max(mach_end))

    crit_job, crit_op = -1, -1
    for j in range(nj):
        if abs(job_end[j] - makespan) < 1e-9:
            crit_job = j
            crit_op = int(job_op[j]) - 1
            break
    if crit_job == -1:
        return []

    visited = np.zeros(n, dtype=np.int8)
    path = []
    while True:
        idx = int(job_to_idx[crit_job, crit_op])
        if idx == -1 or visited[idx]:
            break
        visited[idx] = 1
        path.append((crit_job, crit_op))
        pj = bool(pred_j[idx])
        pm = bool(pred_m[idx])
        if crit_op > 0 and pj:
            crit_op -= 1
        else:
            found = False
            cur_m = int(sched_m[idx])
            cur_s = float(sched_s[idx])
            if pm:
                for idx2 in range(n):
                    if sched_m[idx2] == cur_m and abs(sched_e[idx2] - cur_s) < 1e-9:
                        cj2 = int(seq_arr[idx2])
                        for o2 in range(nm):
                            if job_to_idx[cj2, o2] == idx2:
                                crit_job = cj2
                                crit_op = o2
                                found = True
                                break
                        if found:
                            break
            if not found:
                break
    return path


def run(env_data, state, calc_makespan_fn):
    seq = list(state.sequence)
    crit_path = _extract_critical_path(seq, env_data)
    if not crit_path:
        return state

    # Build mapping from (job, op) to sequence position
    counts = {}
    job_op_to_idx = {}
    for idx, job in enumerate(seq):
        op = counts.get(job, 0)
        job_op_to_idx[(job, op)] = idx
        counts[job] = op + 1

    crit_indices = [job_op_to_idx.get(node) for node in crit_path if node in job_op_to_idx]

    best_seq, best_ms = seq[:], state.makespan
    for _ in range(20):
        if random.random() < 0.7 and len(crit_indices) >= 2:
            i, j = random.sample(crit_indices, 2)
            ns = seq[:]
            ns[i], ns[j] = ns[j], ns[i]
        elif len(crit_indices) >= 3:
            blen = random.randint(2, 3)
            start = random.randint(0, len(crit_indices) - blen)
            b_idx = crit_indices[start : start + blen]
            block = [seq[k] for k in b_idx]
            rem = [v for k, v in enumerate(seq) if k not in b_idx]
            pos = random.randint(0, len(rem))
            ns = rem[:pos] + block + rem[pos:]
        else:
            i, j = random.sample(range(len(seq)), 2)
            ns = seq[:]
            ns[i], ns[j] = ns[j], ns[i]
        ms = calc_makespan_fn(ns, env_data)
        if ms < best_ms:
            best_ms, best_seq = ms, ns[:]

    state.sequence, state.makespan = best_seq, best_ms
    return state
