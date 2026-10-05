import random
import numpy as np
import domain_evaluator


def run(env_data, state, calc_makespan_fn):
    seq = state.sequence[:]
    alloc = state.metadata["machine_alloc"].copy()

    num_jobs = env_data["num_jobs"]
    num_ops = env_data["num_ops"]
    num_machines = env_data["num_machines"]
    processing_matrix = env_data["processing_matrix"]

    # 1. Compute current workload of each machine
    machine_load = np.zeros(num_machines, dtype=np.int32)
    for i in range(num_jobs):
        for j in range(num_ops[i]):
            m = alloc[i, j]
            if m >= 0:
                machine_load[m] += processing_matrix[i, j, m]

    # Find the machine with the minimum workload
    m_idle = int(np.argmin(machine_load))

    # Find all operations that can be processed on m_idle but are currently assigned elsewhere
    candidates = []
    for i in range(num_jobs):
        for j in range(num_ops[i]):
            current_m = alloc[i, j]
            if current_m != m_idle and processing_matrix[i, j, m_idle] >= 0:
                candidates.append((i, j))

    if candidates:
        job_id, op_id = random.choice(candidates)
        alloc[job_id, op_id] = m_idle
    else:
        # Fallback to random reassign
        fallback_candidates = []
        for i in range(num_jobs):
            for j in range(num_ops[i]):
                p_times = processing_matrix[i, j]
                valid = np.where(p_times >= 0)[0]
                if len(valid) > 1:
                    fallback_candidates.append((i, j, valid))
        if fallback_candidates:
            job_id, op_id, valid_machines = random.choice(fallback_candidates)
            current_mac = alloc[job_id, op_id]
            choices = [m for m in valid_machines if m != current_mac]
            if choices:
                alloc[job_id, op_id] = random.choice(choices)

    state.sequence = seq
    state.metadata["machine_alloc"] = alloc

    domain_evaluator.set_active_machine_allocation(alloc)
    state.makespan = calc_makespan_fn(seq, env_data)
    return state
