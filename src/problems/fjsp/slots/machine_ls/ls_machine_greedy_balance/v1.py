import numpy as np
import domain_evaluator


def run(env_data, state, calc_makespan_fn):
    seq = list(state.sequence)
    # Defensive conversion to 2D numpy array to prevent list-indexing tuple error
    alloc = np.ascontiguousarray(np.array(state.metadata["machine_alloc"], dtype=np.int32))

    num_jobs = env_data["num_jobs"]
    num_machines = env_data["num_machines"]
    num_ops = env_data["num_ops"]
    processing_matrix = env_data["processing_matrix"]

    # 1. Compute workloads of machines
    machine_load = np.zeros(num_machines, dtype=np.int32)
    for i in range(num_jobs):
        for j in range(num_ops[i]):
            m = alloc[i, j]
            if m >= 0 and m < num_machines:
                pt = processing_matrix[i, j, m]
                if pt >= 0:
                    machine_load[m] += pt

    avg_load = machine_load.mean()
    if avg_load == 0:
        return state

    # 2. Identify overloaded and underloaded machines
    overloaded = int(np.argmax(machine_load))
    underloaded = int(np.argmin(machine_load))

    if overloaded == underloaded:
        return state

    # 3. Try transferring an operation from the overloaded machine to the underloaded machine
    best_alloc = alloc.copy()
    best_ms = state.makespan
    improved = False

    for i in range(num_jobs):
        for j in range(num_ops[i]):
            if alloc[i, j] == overloaded:
                # Check if the underloaded machine can process this operation
                p_times = processing_matrix[i, j]
                if p_times[underloaded] >= 0:
                    trial_alloc = alloc.copy()
                    trial_alloc[i, j] = underloaded

                    domain_evaluator.set_active_machine_allocation(trial_alloc)
                    ms = domain_evaluator.evaluate_trial(state, seq, trial_alloc, env_data, calc_makespan_fn)
                    if ms < best_ms:
                        best_ms = ms
                        best_alloc = trial_alloc
                        improved = True
                        break  # First-improvement step
        if improved:
            break

    state.sequence = seq
    state.metadata["machine_alloc"] = best_alloc

    domain_evaluator.set_active_machine_allocation(best_alloc)
    state.makespan = best_ms
    return state
