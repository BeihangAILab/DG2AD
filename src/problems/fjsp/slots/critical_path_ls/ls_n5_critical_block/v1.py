import numpy as np
import domain_evaluator


def run(env_data, state, calc_makespan_fn):
    seq = list(state.sequence)
    alloc = state.metadata["machine_alloc"].copy()
    len(seq)

    num_jobs = env_data["num_jobs"]
    num_machines = env_data["num_machines"]
    num_ops = env_data["num_ops"]
    processing_matrix = env_data["processing_matrix"]

    # Bind current machine allocation to evaluator
    domain_evaluator.set_active_machine_allocation(alloc)

    # 1. Simulate scheduling to find completion times of operations
    job_op_idx = np.zeros(num_jobs, dtype=np.int32)
    job_avail = np.zeros(num_jobs, dtype=np.int32)
    mach_avail = np.zeros(num_machines, dtype=np.int32)
    op_end_times = {}  # (job_id, op_idx) -> end_time

    for job_id in seq:
        op_idx = job_op_idx[job_id]
        if op_idx >= num_ops[job_id]:
            continue
        m = alloc[job_id, op_idx]
        if m < 0 or m >= num_machines:
            continue
        pt = processing_matrix[job_id, op_idx, m]
        if pt < 0:
            continue

        start = max(job_avail[job_id], mach_avail[m])
        end = start + pt
        job_avail[job_id] = end
        mach_avail[m] = end
        op_end_times[(job_id, op_idx)] = end
        job_op_idx[job_id] += 1

    makespan = int(np.max(mach_avail)) if len(mach_avail) > 0 else 0
    if makespan == 0:
        return state

    # 2. Identify bottleneck machine and its critical operations
    bottleneck_mach = int(np.argmax(mach_avail))
    critical_ops = []
    for (job_id, op_idx), end_time in op_end_times.items():
        if alloc[job_id, op_idx] == bottleneck_mach:
            critical_ops.append((job_id, op_idx, end_time))

    if len(critical_ops) < 2:
        return state

    # Sort critical operations by their end times
    critical_ops.sort(key=lambda x: x[2])

    # 3. N5 block neighborhood local search: Swap adjacent operations of different jobs
    best_seq = seq[:]
    best_ms = state.makespan

    for k in range(len(critical_ops) - 1):
        job_a, op_a, _ = critical_ops[k]
        job_b, op_b, _ = critical_ops[k + 1]

        if job_a == job_b:
            continue

        trial_seq = seq[:]
        pos_a = None
        pos_b = None
        count_a = 0
        count_b = 0
        for idx, jid in enumerate(trial_seq):
            if jid == job_a:
                count_a += 1
                if count_a == op_a + 1:
                    pos_a = idx
            if jid == job_b:
                count_b += 1
                if count_b == op_b + 1:
                    pos_b = idx

        if pos_a is not None and pos_b is not None:
            trial_seq[pos_a], trial_seq[pos_b] = trial_seq[pos_b], trial_seq[pos_a]
            ms = calc_makespan_fn(trial_seq, env_data)
            if ms < best_ms:
                best_ms = ms
                best_seq = trial_seq[:]

    state.sequence = best_seq
    state.metadata["machine_alloc"] = alloc

    state.makespan = best_ms
    return state
