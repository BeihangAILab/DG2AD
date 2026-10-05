import random
import numpy as np
import domain_evaluator


def run(env_data, state, calc_makespan_fn):
    seq = list(state.sequence)
    # Defensive conversion to 2D numpy array
    alloc = np.ascontiguousarray(np.array(state.metadata["machine_alloc"], dtype=np.int32))

    num_jobs = env_data["num_jobs"]
    num_ops = env_data["num_ops"]
    processing_matrix = env_data["processing_matrix"]
    n = len(seq)

    if n < 2:
        return state

    best_seq = seq[:]
    best_alloc = alloc.copy()
    best_ms = state.makespan

    domain_evaluator.set_active_machine_allocation(best_alloc)

    # Variable Neighborhood Descent (VND) loop
    vnd_improved = True
    max_iterations = 5  # Limit VNS depth to prevent excessive execution time
    iteration = 0

    while vnd_improved and iteration < max_iterations:
        vnd_improved = False
        iteration += 1

        # 1. Neighborhood 1: Machine Reassignment (try 10 random operations)
        candidates = [(i, j) for i in range(num_jobs) for j in range(num_ops[i])]
        random.shuffle(candidates)

        n1_improved = False
        for job_id, op_id in candidates[:10]:
            p_times = processing_matrix[job_id, op_id]
            valid_machines = np.where(p_times >= 0)[0]
            current_m = best_alloc[job_id, op_id]
            choices = [m for m in valid_machines if m != current_m]

            for m in choices:
                trial_alloc = best_alloc.copy()
                trial_alloc[job_id, op_id] = m
                domain_evaluator.set_active_machine_allocation(trial_alloc)
                ms = domain_evaluator.evaluate_trial(state, best_seq, trial_alloc, env_data, calc_makespan_fn)
                if ms < best_ms:
                    best_ms = ms
                    best_alloc = trial_alloc
                    n1_improved = True
                    vnd_improved = True
                    break
            if n1_improved:
                break

        if n1_improved:
            continue  # Restart VND loop with Neighborhood 1

        # 2. Neighborhood 2: Adjacent Sequence Swap (try 10 random adjacent pairs of different jobs)
        valid_indices = [idx for idx in range(n - 1) if best_seq[idx] != best_seq[idx + 1]]
        if valid_indices:
            random.shuffle(valid_indices)
            for idx in valid_indices[:10]:
                trial_seq = best_seq[:]
                trial_seq[idx], trial_seq[idx + 1] = trial_seq[idx + 1], trial_seq[idx]
                domain_evaluator.set_active_machine_allocation(best_alloc)
                ms = domain_evaluator.evaluate_trial(state, trial_seq, best_alloc, env_data, calc_makespan_fn)
                if ms < best_ms:
                    best_ms = ms
                    best_seq = trial_seq[:]
                    vnd_improved = True
                    break

    state.sequence = best_seq
    state.metadata["machine_alloc"] = best_alloc

    domain_evaluator.set_active_machine_allocation(best_alloc)
    state.makespan = best_ms
    return state
