import random


def run(env_data, state, calc_makespan_fn):
    if not hasattr(state, "metadata") or state.metadata is None:
        state.metadata = {}

    if "elite_seq" not in state.metadata:
        state.metadata["elite_seq"] = state.sequence[:]
        state.metadata["elite_ms"] = state.makespan
        return state

    if state.makespan < state.metadata["elite_ms"]:
        state.metadata["elite_seq"] = state.sequence[:]
        state.metadata["elite_ms"] = state.makespan

    elite_seq = state.metadata["elite_seq"]
    curr_seq = state.sequence[:]

    # Very simple crossover: pick a route from elite and prepend it, remove duplicates
    elite_routes = []
    curr = []
    for x in elite_seq:
        if x == 0:
            if curr:
                elite_routes.append(curr)
            curr = []
        else:
            curr.append(x)
    if curr:
        elite_routes.append(curr)

    if not elite_routes:
        return state
    r = random.choice(elite_routes)

    new_seq = []
    for x in r:
        new_seq.append(x)
    new_seq.append(0)

    r_set = set(r)
    for x in curr_seq:
        if x == 0 and (not new_seq or new_seq[-1] != 0):
            new_seq.append(0)
        elif x != 0 and x not in r_set:
            new_seq.append(x)

    while new_seq and new_seq[-1] == 0:
        new_seq.pop()

    ms = calc_makespan_fn(new_seq, env_data)
    if ms < 999999999:
        state.sequence = new_seq
        state.makespan = ms

    return state
