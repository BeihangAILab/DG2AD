import numpy as np


def run(env_data: dict, state, calc_makespan_fn):
    num_jobs = env_data["num_jobs"]
    num_machines = env_data["num_machines"]
    pt = env_data["processing_times"]  # shape=(num_machines, num_jobs)

    # --- Randomized NEH heuristic ---
    # Compute total processing time for each job
    total_pt = np.sum(pt, axis=0)  # shape=(num_jobs,)

    # Sort jobs by descending total processing time, but add randomness
    noise = np.random.uniform(0, 0.3) * np.std(total_pt)
    perturbed_pt = total_pt + np.random.normal(0, max(noise, 1.0), size=num_jobs)
    job_order = np.argsort(-perturbed_pt).tolist()

    # NEH: insert jobs one by one into the best position
    # We work with job-level permutation, then convert to operation-based encoding
    partial = [job_order[0]]

    def perm_to_seq(perm):
        seq = np.repeat(np.array(perm, dtype=np.int32), num_machines)
        return seq

    def eval_perm(perm):
        state.sequence = perm_to_seq(perm)
        return calc_makespan_fn(state, env_data)

    for i in range(1, num_jobs):
        job = job_order[i]
        best_mk = float("inf")
        best_positions = []

        for pos in range(len(partial) + 1):
            candidate = partial[:pos] + [job] + partial[pos:]
            mk = eval_perm(candidate)
            if mk < best_mk:
                best_mk = mk
                best_positions = [pos]
            elif mk == best_mk:
                best_positions.append(pos)

        # Randomly pick among tied best positions for diversity
        chosen_pos = best_positions[np.random.randint(len(best_positions))]
        partial = partial[:chosen_pos] + [job] + partial[chosen_pos:]

    best_perm = partial[:]
    best_seq = perm_to_seq(best_perm)
    state.sequence = best_seq.copy()
    best_mk = calc_makespan_fn(state, env_data)
    state.makespan = best_mk

    # --- Local search: pairwise insertion on the job permutation ---
    improved = True
    max_iters = 3
    iteration = 0
    while improved and iteration < max_iters:
        improved = False
        iteration += 1
        # Random order of jobs to try removing
        indices = list(range(num_jobs))
        np.random.shuffle(indices)
        for idx in indices:
            job = best_perm[idx]
            reduced = best_perm[:idx] + best_perm[idx + 1 :]
            found_better = False
            for pos in range(num_jobs):
                if pos == idx:
                    continue
                candidate = reduced[:pos] + [job] + reduced[pos:]
                mk = eval_perm(candidate)
                if mk < best_mk:
                    best_mk = mk
                    best_perm = candidate
                    best_seq = perm_to_seq(candidate)
                    improved = True
                    found_better = True
                    break
            if found_better:
                break

    state.sequence = best_seq.copy()
    state.makespan = best_mk
    state.metadata = {}
    return state
