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

    demands = env_data["demands"]
    cap = env_data["capacity"]

    while improved:
        improved = False
        for r1 in range(len(routes)):
            for r2 in range(r1 + 1, len(routes)):
                for i in range(1, len(routes[r1])):
                    for j in range(1, len(routes[r2])):
                        load1 = sum(demands[x] for x in routes[r1][:i]) + sum(
                            demands[x] for x in routes[r2][j:]
                        )
                        load2 = sum(demands[x] for x in routes[r2][:j]) + sum(
                            demands[x] for x in routes[r1][i:]
                        )
                        if load1 <= cap and load2 <= cap:
                            new_r1 = routes[r1][:i] + routes[r2][j:]
                            new_r2 = routes[r2][:j] + routes[r1][i:]
                            if not new_r1 or not new_r2:
                                continue
                            new_routes = routes[:]
                            new_routes[r1] = new_r1
                            new_routes[r2] = new_r2
                            new_seq = []
                            for r in new_routes:
                                new_seq.extend(r)
                                new_seq.append(0)
                            new_seq.pop()
                            ms = calc_makespan_fn(new_seq, env_data)
                            if ms < best_ms:
                                best_ms = ms
                                routes = new_routes
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
