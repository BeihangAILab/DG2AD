import random


def run(env_data, state, calc_makespan_fn):
    nj, nm = env_data["num_jobs"], env_data["num_machines"]
    base = [j for j in range(nj) for _ in range(nm)]
    seq = random.sample(base, len(base))
    state.sequence = seq
    state.makespan = calc_makespan_fn(seq, env_data)
    state.metadata.update(
        {"g_best_ms": state.makespan, "g_best_seq": seq[:], "no_improve": 0, "temp": 30.0}
    )
    return state
