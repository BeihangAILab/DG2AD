import random
import numpy as np
import domain_evaluator


def run(env_data, state, calc_makespan_fn):
    num_jobs = env_data["num_jobs"]
    max_ops = env_data["max_operations"]
    processing_matrix = env_data["processing_matrix"]
    num_ops = env_data["num_ops"]

    # 1. ALYS Machine Alloc: Select the machine with the minimum processing time
    machine_alloc = np.zeros((num_jobs, max_ops), dtype=np.int32)
    for i in range(num_jobs):
        for j in range(max_ops):
            p_times = processing_matrix[i, j]
            valid_machines = np.where(p_times >= 0)[0]
            if len(valid_machines) > 0:
                # Select the machine that has the minimum processing time
                min_time = np.min(p_times[valid_machines])
                best_machines = valid_machines[p_times[valid_machines] == min_time]
                machine_alloc[i, j] = random.choice(best_machines)
            else:
                machine_alloc[i, j] = -1

    # 2. Random OBR Sequence
    seq = [job_id for job_id in range(num_jobs) for _ in range(num_ops[job_id])]
    random.shuffle(seq)

    state.sequence = seq
    state.metadata["machine_alloc"] = machine_alloc

    # Bind current machine allocation
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
