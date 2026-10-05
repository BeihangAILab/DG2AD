import random
import numpy as np
import domain_evaluator


def run(env_data, state, calc_makespan_fn):
    seq = list(state.sequence)
    # Defensive conversion to 2D numpy array
    alloc = np.ascontiguousarray(np.array(state.metadata["machine_alloc"], dtype=np.int32))
    n = len(seq)

    if n < 2:
        return state

    num_jobs = env_data["num_jobs"]
    num_ops = env_data["num_ops"]
    processing_matrix = env_data["processing_matrix"]

    meta = state.metadata
    tabu_list = list(meta.get("tabu_list", []))
    tabu_tenure = 7  # Tabu tenure

    seq[:]
    alloc.copy()

    candidates = []

    # Candidate 1: Adjacent sequence swap (only swap different jobs)
    for _ in range(5):
        i = random.randint(0, n - 2)
        if seq[i] == seq[i + 1]:
            continue
        move_key = ("swap", i, i + 1)
        if move_key not in tabu_list:
            trial_seq = seq[:]
            trial_seq[i], trial_seq[i + 1] = trial_seq[i + 1], trial_seq[i]

            # Explicitly bind correct machine alloc before evaluation
            domain_evaluator.set_active_machine_allocation(alloc)
            ms = domain_evaluator.evaluate_trial(state, trial_seq, alloc, env_data, calc_makespan_fn)
            candidates.append((ms, trial_seq[:], alloc.copy(), move_key))

    # Candidate 2: Machine reassignment
    for _ in range(5):
        job_id = random.randint(0, num_jobs - 1)
        op_id = random.randint(0, num_ops[job_id] - 1)
        p_times = processing_matrix[job_id, op_id]
        valid_machines = np.where(p_times >= 0)[0]
        if len(valid_machines) > 1:
            current_m = alloc[job_id, op_id]
            choices = [m for m in valid_machines if m != current_m]
            if choices:
                new_m = random.choice(choices)
                move_key = ("machine", job_id, op_id, new_m)
                if move_key not in tabu_list:
                    trial_alloc = alloc.copy()
                    trial_alloc[job_id, op_id] = new_m

                    # Explicitly bind correct machine alloc before evaluation
                    domain_evaluator.set_active_machine_allocation(trial_alloc)
                    ms = domain_evaluator.evaluate_trial(state, seq, trial_alloc, env_data, calc_makespan_fn)
                    candidates.append((ms, seq[:], trial_alloc, move_key))

    # Candidate 3: Sequence insertion
    for _ in range(5):
        pos_from = random.randint(0, n - 1)
        pos_to = random.randint(0, n - 1)
        if pos_from == pos_to:
            continue
        move_key = ("insert", pos_from, pos_to)
        if move_key not in tabu_list:
            trial_seq = seq[:]
            elem = trial_seq.pop(pos_from)
            trial_seq.insert(pos_to, elem)

            # Explicitly bind correct machine alloc before evaluation
            domain_evaluator.set_active_machine_allocation(alloc)
            ms = domain_evaluator.evaluate_trial(state, trial_seq, alloc, env_data, calc_makespan_fn)
            candidates.append((ms, trial_seq[:], alloc.copy(), move_key))

    if not candidates:
        # If all moves are tabu, clear tabu list and keep current state
        tabu_list = []
        meta["tabu_list"] = tabu_list
        return state

    # Select the best candidate
    candidates.sort(key=lambda x: x[0])
    best_candidate = candidates[0]
    candidate_ms, candidate_seq, candidate_alloc, move_key = best_candidate

    # Apply the best candidate (allows degradation to escape local minima)
    state.sequence = candidate_seq
    state.metadata["machine_alloc"] = candidate_alloc
    domain_evaluator.set_active_machine_allocation(candidate_alloc)
    state.makespan = candidate_ms

    # Update tabu list
    tabu_list.append(move_key)
    if len(tabu_list) > tabu_tenure:
        tabu_list = tabu_list[-tabu_tenure:]

    meta["tabu_list"] = tabu_list
    return state
