import random
import numpy as np
import domain_evaluator


def run(env_data, state, calc_makespan_fn):
    seq = list(state.sequence)
    # Defensive NumPy array conversion
    alloc = np.ascontiguousarray(np.array(state.metadata["machine_alloc"], dtype=np.int32))

    num_jobs = env_data["num_jobs"]
    num_ops = env_data["num_ops"]
    processing_matrix = env_data["processing_matrix"]

    # 1. Randomly choose 20% of the jobs to have their machine allocations reset
    num_reset_jobs = max(1, int(num_jobs * 0.2))
    reset_jobs = random.sample(range(num_jobs), num_reset_jobs)

    for job_id in reset_jobs:
        for op_idx in range(num_ops[job_id]):
            p_times = processing_matrix[job_id, op_idx]
            valid_machines = np.where(p_times >= 0)[0]
            if len(valid_machines) > 0:
                alloc[job_id, op_idx] = random.choice(valid_machines)

    # 2. Shuffle the sequence for further diversification
    random.shuffle(seq)

    state.sequence = seq
    state.metadata["machine_alloc"] = alloc

    domain_evaluator.set_active_machine_allocation(alloc)
    state.makespan = calc_makespan_fn(seq, env_data)
    return state
