import numpy as np
import domain_evaluator


def run(env_data, state, calc_makespan_fn):
    seq = list(state.sequence)
    alloc = state.metadata["machine_alloc"].copy()

    num_jobs = env_data["num_jobs"]
    num_machines = env_data["num_machines"]
    num_ops = env_data["num_ops"]
    processing_matrix = env_data["processing_matrix"]

    # 1. Simulate to find machine workloads and completion times
    job_op_idx = np.zeros(num_jobs, dtype=np.int32)
    job_avail = np.zeros(num_jobs, dtype=np.int32)
    mach_avail = np.zeros(num_machines, dtype=np.int32)
    machine_load = np.zeros(num_machines, dtype=np.int32)
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
        machine_load[m] += pt
        op_end_times[(job_id, op_idx)] = end
        job_op_idx[job_id] += 1

    makespan = int(np.max(mach_avail)) if len(mach_avail) > 0 else 0
    if makespan == 0:
        return state

    # 2. Identify bottleneck machine and critical operations
    bottleneck_mach = int(np.argmax(mach_avail))
    critical_candidates = []
    for (job_id, op_idx), end_time in op_end_times.items():
        if alloc[job_id, op_idx] == bottleneck_mach:
            pt = processing_matrix[job_id, op_idx, bottleneck_mach]
            critical_candidates.append((job_id, op_idx, pt, end_time))

    if not critical_candidates:
        return state

    # Sort critical operations by end time descending (most critical first)
    critical_candidates.sort(key=lambda x: x[3], reverse=True)

    best_alloc = alloc.copy()
    best_ms = state.makespan

    # 3. Greedy machine reassignment
    for job_id, op_idx, pt_on_bottleneck, _ in critical_candidates[:5]:
        p_times = processing_matrix[job_id, op_idx]
        valid_machines = np.where(p_times >= 0)[0]
        current_m = alloc[job_id, op_idx]

        choices = [m for m in valid_machines if m != current_m]
        if not choices:
            continue

        # Sort choices by workload after assignment to target the most idle machines first
        choices.sort(key=lambda m: machine_load[m] + p_times[m])

        for m in choices:
            trial_alloc = alloc.copy()
            trial_alloc[job_id, op_idx] = m

            domain_evaluator.set_active_machine_allocation(trial_alloc)
            ms = domain_evaluator.evaluate_trial(state, seq, trial_alloc, env_data, calc_makespan_fn)
            if ms < best_ms:
                best_ms = ms
                best_alloc = trial_alloc
                alloc = trial_alloc.copy()  # Accumulate improvement
                # Update workload tracking
                machine_load[current_m] = max(0, machine_load[current_m] - pt_on_bottleneck)
                machine_load[m] += p_times[m]
                current_m = m
                break

    state.sequence = seq
    state.metadata["machine_alloc"] = best_alloc

    domain_evaluator.set_active_machine_allocation(best_alloc)
    state.makespan = best_ms
    return state
