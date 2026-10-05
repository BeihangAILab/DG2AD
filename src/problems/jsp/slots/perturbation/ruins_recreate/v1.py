import random


def run(env_data, state, calc_makespan_fn):
    nj, nm, seq = env_data["num_jobs"], env_data["num_machines"], list(state.sequence)
    ruined = random.sample(range(nj), 2)
    ns = [j for j in seq if j not in ruined]
    for job in ruined:
        for _ in range(nm):
            b_p, b_v = 0, float("inf")
            for p in random.sample(range(len(ns) + 1), min(10, len(ns) + 1)):
                ts = ns[:p] + [job] + ns[p:]
                v = calc_makespan_fn(ts, env_data)
                if v < b_v:
                    b_v, b_p = v, p
            ns.insert(b_p, job)
    state.sequence, state.makespan = ns, calc_makespan_fn(ns, env_data)
    return state
