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

    if n < 3:
        return state

    # 1. Ruin: Randomly remove 5% of operations from the sequence
    ruin_count = max(1, int(n * 0.05))
    ruin_indices = sorted(random.sample(range(n), ruin_count), reverse=True)

    removed_items = []
    for idx in ruin_indices:
        removed_items.append(seq.pop(idx))

    # 2. Recreate: Append the removed items to the end of the sequence
    random.shuffle(removed_items)
    seq.extend(removed_items)

    # 3. Greedy Machine Reallocation for affected jobs
    num_machines = env_data["num_machines"]
    machine_load = np.zeros(num_machines, dtype=np.int32)
    for i in range(num_jobs):
        for j in range(num_ops[i]):
            m = alloc[i, j]
            if m >= 0:
                machine_load[m] += processing_matrix[i, j, m]

    for job_id in set(removed_items):
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
