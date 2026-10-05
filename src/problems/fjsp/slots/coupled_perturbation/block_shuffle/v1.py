import random
import numpy as np
import domain_evaluator


def run(env_data, state, calc_makespan_fn):
    seq = list(state.sequence)
    alloc = state.metadata["machine_alloc"].copy()

    num_jobs = env_data["num_jobs"]
    processing_matrix = env_data["processing_matrix"]
    num_ops = env_data["num_ops"]
    n = len(seq)

    if n < 4:
        return state

    # 1. Select block of size 5
    block_size = min(5, n - 1)
    if block_size < 2:
        return state
    block_start = random.randint(0, n - block_size)

    # 2. Shuffle inside the block
    block = seq[block_start : block_start + block_size]
    shuffled = block[:]
    for _ in range(10):
        random.shuffle(shuffled)
        if shuffled != block:
            break
    seq[block_start : block_start + block_size] = shuffled

    # 3. Greedy machine balancing for the jobs in the block
    num_machines = env_data["num_machines"]
    machine_load = np.zeros(num_machines, dtype=np.int32)
    for i in range(num_jobs):
        for j in range(num_ops[i]):
            m = alloc[i, j]
            if m >= 0:
                machine_load[m] += processing_matrix[i, j, m]

    for job_id in set(block):
        for op_idx in range(num_ops[job_id]):
            p_times = processing_matrix[job_id, op_idx]
            valid_machines = np.where(p_times >= 0)[0]
            if len(valid_machines) > 0:
                old_m = alloc[job_id, op_idx]
                if old_m >= 0:
                    machine_load[old_m] = max(0, machine_load[old_m] - p_times[old_m])

                best_m = valid_machines[0]
                min_load_after = machine_load[best_m] + p_times[best_m]
                for m in valid_machines:
                    load_after = machine_load[m] + p_times[m]
                    if load_after < min_load_after:
                        min_load_after = load_after
                        best_m = m
                alloc[job_id, op_idx] = best_m
                machine_load[best_m] += p_times[best_m]

    state.sequence = seq
    state.metadata["machine_alloc"] = alloc

    domain_evaluator.set_active_machine_allocation(alloc)
    state.makespan = calc_makespan_fn(seq, env_data)
    return state
