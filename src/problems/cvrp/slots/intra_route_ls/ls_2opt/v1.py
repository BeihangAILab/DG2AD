def run(env_data, state, calc_makespan_fn):
    seq = state.sequence[:]
    best_ms = state.makespan
    improved = True

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

    while improved:
        improved = False
        for r_idx, route in enumerate(routes):
            if len(route) < 3:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route)):
                    new_route = route[:i] + route[i:j][::-1] + route[j:]
                    new_routes = routes[:r_idx] + [new_route] + routes[r_idx + 1 :]
                    new_seq = []
                    for r in new_routes:
                        new_seq.extend(r)
                        new_seq.append(0)
                    new_seq.pop()
                    ms = calc_makespan_fn(new_seq, env_data)
                    if ms < best_ms:
                        best_ms = ms
                        routes[r_idx] = new_route
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    new_seq = []
    for r in routes:
        new_seq.extend(r)
        new_seq.append(0)
    if new_seq:
        new_seq.pop()

    state.sequence = new_seq
    state.makespan = best_ms
    return state
