import random


def run(env_data, state, calc_makespan_fn):
    seq = state.sequence[:]
    dist = env_data["dist_matrix"]
    depot = env_data["depot_idx"]

    routes = []
    curr = []
    for x in seq:
        if x == 0:
            if curr:
                routes.append(curr)
            curr = []
        else:
            curr.append(x)
    if curr:
        routes.append(curr)

    if len(routes) < 2:
        return state

    # Find bottleneck route
    max_d = -1
    bottleneck_idx = -1
    for i, r in enumerate(routes):
        d = 0
        prev = depot
        for c in r:
            d += dist[prev, c]
            prev = c
        d += dist[prev, depot]
        if d > max_d:
            max_d = d
            bottleneck_idx = i

    if bottleneck_idx == -1 or len(routes[bottleneck_idx]) == 0:
        return state

    route = routes[bottleneck_idx]
    cust = random.choice(route)
    route.remove(cust)
    if not route:
        routes.pop(bottleneck_idx)

    # Reinsert randomly into another route
    target_idx = random.choice([i for i in range(len(routes)) if i != bottleneck_idx] or [0])
    ins_pos = random.randint(0, len(routes[target_idx]))
    routes[target_idx].insert(ins_pos, cust)

    new_seq = []
    for r in routes:
        new_seq.extend(r)
        new_seq.append(0)
    if new_seq:
        new_seq.pop()

    ms = calc_makespan_fn(new_seq, env_data)
    if ms < 999999999:
        state.sequence = new_seq
        state.makespan = ms

    return state
