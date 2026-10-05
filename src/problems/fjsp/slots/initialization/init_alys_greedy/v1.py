import random
import numpy as np
import domain_evaluator


def run(env_data, state, calc_makespan_fn):
    num_jobs = env_data["num_jobs"]
    max_ops = env_data["max_operations"]
    processing_matrix = env_data["processing_matrix"]
    num_ops = env_data["num_ops"]

    # 1. ALYS Machine Alloc: Choose machine with shortest processing time
    machine_alloc = np.zeros((num_jobs, max_ops), dtype=np.int32)
    for i in range(num_jobs):
        for j in range(max_ops):
            p_times = processing_matrix[i, j]
            valid_machines = np.where(p_times >= 0)[0]
            if len(valid_machines) > 0:
                min_time = np.min(p_times[valid_machines])
                best_machines = valid_machines[p_times[valid_machines] == min_time]
                machine_alloc[i, j] = random.choice(best_machines)
            else:
                machine_alloc[i, j] = -1

    # 2. MWKR (Most Work Remaining) Sequence Generation
    op_idx = np.zeros(num_jobs, dtype=np.int32)
    job_work_remaining = np.zeros(num_jobs, dtype=np.int32)
    for i in range(num_jobs):
        total_w = 0
        for j in range(num_ops[i]):
            m = machine_alloc[i, j]
            if m >= 0:
                total_w += processing_matrix[i, j, m]
        job_work_remaining[i] = total_w

    seq = []
    total_ops = int(np.sum(num_ops))
    while len(seq) < total_ops:
        best_job = -1
        max_work = -1
        for i in range(num_jobs):
            if op_idx[i] < num_ops[i]:
                w = job_work_remaining[i]
                if w > max_work:
                    max_work = w
                    best_job = i
        if best_job == -1:
            for i in range(num_jobs):
                if op_idx[i] < num_ops[i]:
                    best_job = i
                    break
        seq.append(best_job)
        m_chosen = machine_alloc[best_job, op_idx[best_job]]
        if m_chosen >= 0:
            p_time = processing_matrix[best_job, op_idx[best_job], m_chosen]
            job_work_remaining[best_job] -= p_time
        op_idx[best_job] += 1

    state.sequence = seq
    state.metadata["machine_alloc"] = machine_alloc

    domain_evaluator.set_active_machine_allocation(machine_alloc)
    state.makespan = calc_makespan_fn(seq, env_data)

    state.metadata.update(
        {
            "g_best_ms": state.makespan,
            "g_best_seq": seq[:],
            "g_best_alloc": machine_alloc.copy(),
            "prev_alloc": machine_alloc.copy(),
            "no_improve": 0,
            "temp": 30.0,
        }
    )
    return state
