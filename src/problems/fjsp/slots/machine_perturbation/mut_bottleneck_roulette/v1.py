import random
import numpy as np
import domain_evaluator


def run(env_data, state, calc_makespan_fn):
    seq = state.sequence[:]
    alloc = state.metadata["machine_alloc"].copy()

    num_jobs = env_data["num_jobs"]
    num_machines = env_data["num_machines"]
    processing_matrix = env_data["processing_matrix"]
    num_ops = env_data["num_ops"]

    # 1. Estimate workload of each machine
    machine_load = np.zeros(num_machines, dtype=np.int32)
    for i in range(num_jobs):
        for j in range(num_ops[i]):
            m = alloc[i, j]
            if m >= 0 and m < num_machines:
                pt = processing_matrix[i, j, m]
                if pt >= 0:
                    machine_load[m] += pt

    # Find the bottleneck machine (maximum workload)
    bottleneck = int(np.argmax(machine_load))

    # 2. Collect operations assigned to the bottleneck machine
    bottleneck_ops = []
    for i in range(num_jobs):
        for j in range(num_ops[i]):
            if alloc[i, j] == bottleneck:
                pt = processing_matrix[i, j, bottleneck]
                if pt >= 0:
                    bottleneck_ops.append((i, j, pt))

    if bottleneck_ops:
        # Roulette selection: operations with longer processing times have a higher probability of being chosen
        weights = [pt for _, _, pt in bottleneck_ops]
        total_w = sum(weights)
        if total_w > 0:
            probs = [w / total_w for w in weights]
            idx = random.choices(range(len(bottleneck_ops)), weights=probs, k=1)[0]
        else:
            idx = random.randint(0, len(bottleneck_ops) - 1)

        job_id, op_id, _ = bottleneck_ops[idx]

        # Reassign to a non-bottleneck machine that has the minimum load after assignment
        p_times = processing_matrix[job_id, op_id]
        valid_machines = np.where(p_times >= 0)[0]
        choices = [m for m in valid_machines if m != bottleneck]
        if choices:
            best_m = min(choices, key=lambda m: machine_load[m] + p_times[m])
            alloc[job_id, op_id] = best_m

    state.sequence = seq
    state.metadata["machine_alloc"] = alloc

    domain_evaluator.set_active_machine_allocation(alloc)
    state.makespan = calc_makespan_fn(seq, env_data)
    return state
