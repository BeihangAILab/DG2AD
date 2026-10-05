import numpy as np
import domain_evaluator


def run(env_data, state, calc_makespan_fn):
    num_jobs = env_data["num_jobs"]
    max_ops = env_data["max_operations"]
    processing_matrix = env_data["processing_matrix"]
    num_ops = env_data["num_ops"]
    num_machines = env_data["num_machines"]

    # 1. Global Machine Load Balancing: Assign operations to minimize the resulting machine load
    machine_alloc = np.zeros((num_jobs, max_ops), dtype=np.int32)
    machine_load = np.zeros(num_machines, dtype=np.int32)

    for j in range(max_ops):
        for i in range(num_jobs):
            if j < num_ops[i]:
                p_times = processing_matrix[i, j]
                valid_machines = np.where(p_times >= 0)[0]
                if len(valid_machines) > 0:
                    best_m = valid_machines[0]
                    min_load_after = machine_load[best_m] + p_times[best_m]
                    for m in valid_machines:
                        load_after = machine_load[m] + p_times[m]
                        if load_after < min_load_after:
                            min_load_after = load_after
                            best_m = m
                    machine_alloc[i, j] = best_m
                    machine_load[best_m] += p_times[best_m]
                else:
                    machine_alloc[i, j] = -1

    # 2. SPT Sequence Generation (Interleaved OBR)
    op_idx = np.zeros(num_jobs, dtype=np.int32)
    seq = []
    total_ops = int(np.sum(num_ops))
    while len(seq) < total_ops:
        best_job = -1
        min_p = 999999999
        for i in range(num_jobs):
            if op_idx[i] < num_ops[i]:
                m = machine_alloc[i, op_idx[i]]
                if m >= 0:
                    p = processing_matrix[i, op_idx[i], m]
                else:
                    p = 999999999
                if p < min_p:
                    min_p = p
                    best_job = i
        if best_job == -1:
            for i in range(num_jobs):
                if op_idx[i] < num_ops[i]:
                    best_job = i
                    break
        seq.append(best_job)
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
