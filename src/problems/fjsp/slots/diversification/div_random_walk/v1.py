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
    n = len(seq)

    if n < 2:
        return state

    # 1. Force 3 machine reassignments (to a different valid machine)
    reassign_count = 0
    attempts = 0
    while reassign_count < 3 and attempts < 20:
        attempts += 1
        job_id = random.randrange(num_jobs)
        op_id = random.randrange(num_ops[job_id])
        p_times = processing_matrix[job_id, op_id]
        valid_machines = np.where(p_times >= 0)[0]
        if len(valid_machines) > 1:
            current_m = alloc[job_id, op_id]
            choices = [m for m in valid_machines if m != current_m]
            if choices:
                alloc[job_id, op_id] = random.choice(choices)
                reassign_count += 1

    # 2. Force 3 sequence swaps of different jobs
    swap_count = 0
    attempts = 0
    while swap_count < 3 and attempts < 20:
        attempts += 1
        i = random.randrange(n)
        different_indices = [j for j in range(n) if seq[j] != seq[i]]
        if different_indices:
            j = random.choice(different_indices)
            seq[i], seq[j] = seq[j], seq[i]
            swap_count += 1

    state.sequence = seq
    state.metadata["machine_alloc"] = alloc

    domain_evaluator.set_active_machine_allocation(alloc)
    state.makespan = calc_makespan_fn(seq, env_data)
    return state
