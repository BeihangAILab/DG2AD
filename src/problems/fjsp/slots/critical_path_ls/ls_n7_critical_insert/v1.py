import numpy as np
import domain_evaluator


def run(env_data, state, calc_makespan_fn):
    seq = list(state.sequence)
    alloc = state.metadata["machine_alloc"].copy()

    num_jobs = env_data["num_jobs"]
    num_machines = env_data["num_machines"]
    num_ops = env_data["num_ops"]
    processing_matrix = env_data["processing_matrix"]

    # Bind current machine allocation
    domain_evaluator.set_active_machine_allocation(alloc)

    # 1. Simulate scheduling to find start/end times of operations
    job_op_idx = np.zeros(num_jobs, dtype=np.int32)
    job_avail = np.zeros(num_jobs, dtype=np.int32)
    mach_avail = np.zeros(num_machines, dtype=np.int32)
    op_start_times = {}
    op_end_times = {}

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
        op_start_times[(job_id, op_idx)] = start
        op_end_times[(job_id, op_idx)] = end
        job_op_idx[job_id] += 1

    makespan = int(np.max(mach_avail)) if len(mach_avail) > 0 else 0
    if makespan == 0:
        return state

    # 2. Identify critical operations on bottleneck machine
    bottleneck_mach = int(np.argmax(mach_avail))
    bottleneck_ops = []
    for (job_id, op_idx), end_time in op_end_times.items():
        if alloc[job_id, op_idx] == bottleneck_mach:
            start = op_start_times[(job_id, op_idx)]
            bottleneck_ops.append((job_id, op_idx, start, end_time))

    # Sort bottleneck operations by start time
    bottleneck_ops.sort(key=lambda x: x[2])

    if len(bottleneck_ops) < 2:
        return state

    # 3. N7 Neighborhood: Insert move on critical operations
    best_seq = seq[:]
    best_ms = state.makespan
    improved = False

    for job_id, op_idx, _, _ in bottleneck_ops[:5]:  # Try top 5 critical ops
        # Locate the position of the (op_idx + 1)-th occurrence of job_id
        count = 0
        current_pos = -1
        for idx, jid in enumerate(seq):
            if jid == job_id:
                count += 1
                if count == op_idx + 1:
                    current_pos = idx
                    break

        if current_pos < 0:
            continue

        element = seq[current_pos]

        # Try inserting at all possible positions in the sequence
        for insert_pos in range(len(seq) + 1):
            if insert_pos == current_pos or insert_pos == current_pos + 1:
                continue

            trial_seq = seq[:]
            trial_seq.pop(current_pos)

            # Adjust insert position after pop
            actual_insert = insert_pos if insert_pos < current_pos else insert_pos - 1
            trial_seq.insert(actual_insert, element)

            ms = calc_makespan_fn(trial_seq, env_data)
            if ms < best_ms:
                best_ms = ms
                best_seq = trial_seq[:]
                improved = True

        if improved:
            # First-improvement step: break out early to prevent excessive computation
            break

    state.sequence = best_seq
    state.metadata["machine_alloc"] = alloc

    state.makespan = best_ms
    return state
