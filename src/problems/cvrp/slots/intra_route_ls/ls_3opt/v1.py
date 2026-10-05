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
            n = len(route)
            if n < 4:
                continue
            for i in range(n - 2):
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n):
                        # 3-opt has 7 alternative combinations, test one simple segment swap
                        p1, p2, p3, p4 = route[:i], route[i:j], route[j:k], route[k:]
                        new_route = p1 + p3 + p2 + p4
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
