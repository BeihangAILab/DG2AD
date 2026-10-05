def run(env_data, state, calc_makespan_fn):
    nj, nm = env_data["num_jobs"], env_data["num_machines"]
    times, machines = env_data["times_matrix"], env_data["machines_matrix"]
    counts, ready_j, ready_m = [0] * nj, [0.0] * nj, [0.0] * nm
    seq = []
    for _ in range(nj * nm):
        candidates = []
        for j in range(nj):
            if counts[j] < nm:
                m, t = int(machines[j][counts[j]]), float(times[j][counts[j]])
                candidates.append((max(ready_j[j], ready_m[m]) + t, j, m, t))
        candidates.sort(key=lambda x: x[0])
        _, cj, cm, ct = candidates[0]
        ready_j[cj] = ready_m[cm] = max(ready_j[cj], ready_m[cm]) + ct
        counts[cj] += 1
        seq.append(cj)
    state.sequence, state.makespan = seq, calc_makespan_fn(seq, env_data)
    return state
