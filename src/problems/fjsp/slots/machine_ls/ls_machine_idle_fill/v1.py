import random
import numpy as np
import domain_evaluator


def run(env_data, state, calc_makespan_fn):
    seq = list(state.sequence)
    # Defensive NumPy conversion to prevent list-indexing errors
    alloc = np.ascontiguousarray(np.array(state.metadata["machine_alloc"], dtype=np.int32))

    num_jobs = env_data["num_jobs"]
    num_machines = env_data["num_machines"]
    num_ops = env_data["num_ops"]
    processing_matrix = env_data["processing_matrix"]

    # 1. Simulate scheduling and record busy intervals
    job_op_idx = np.zeros(num_jobs, dtype=np.int32)
    job_avail = np.zeros(num_jobs, dtype=np.int32)
    mach_busy_intervals = [[] for _ in range(num_machines)]

    def mach_avail_max(m, intervals):
        if not intervals[m]:
            return 0
        return intervals[m][-1][1]

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

        start = max(job_avail[job_id], mach_avail_max(m, mach_busy_intervals))
        end = start + pt
        mach_busy_intervals[m].append((start, end))
        job_avail[job_id] = end
        job_op_idx[job_id] += 1

    makespan = 0
    for m in range(num_machines):
        if mach_busy_intervals[m]:
            end_t = mach_busy_intervals[m][-1][1]
            if end_t > makespan:
                makespan = end_t

    if makespan == 0:
        return state

    # 2. Identify idle time gaps
    idle_gaps = []
    for m in range(num_machines):
        intervals = mach_busy_intervals[m]
        if not intervals:
            continue
        # intervals are already sorted by start time because they are appended in order
        prev_end = 0
        for s, e in intervals:
            if s > prev_end:
                idle_gaps.append((m, prev_end, s))
            prev_end = max(prev_end, e)

    if not idle_gaps:
        return state

    # 3. Try to fill idle gaps by reassigning eligible operations
    best_alloc = alloc.copy()
    best_ms = state.makespan
    improved = False

    random.shuffle(idle_gaps)
    for mach, gap_start, gap_end in idle_gaps[:3]:  # Try up to 3 gaps
        gap_duration = gap_end - gap_start
        if gap_duration <= 0:
            continue

        for i in range(num_jobs):
            for j in range(num_ops[i]):
                if alloc[i, j] == mach:
                    continue
                p_times = processing_matrix[i, j]
                if p_times[mach] >= 0 and p_times[mach] <= gap_duration:
                    trial_alloc = alloc.copy()
                    trial_alloc[i, j] = mach

                    domain_evaluator.set_active_machine_allocation(trial_alloc)
                    ms = domain_evaluator.evaluate_trial(state, seq, trial_alloc, env_data, calc_makespan_fn)
                    if ms < best_ms:
                        best_ms = ms
                        best_alloc = trial_alloc
                        improved = True
                        break
            if improved:
                break
        if improved:
            break

    state.sequence = seq
    state.metadata["machine_alloc"] = best_alloc

    domain_evaluator.set_active_machine_allocation(best_alloc)
    state.makespan = best_ms
    return state
