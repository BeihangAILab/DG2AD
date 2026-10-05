def run(env_data, state, calc_makespan_fn):
    seq = state.sequence[:]
    best_ms = state.makespan
    demands = env_data["demands"]
    cap = env_data["capacity"]

    # Parse routes
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

    if not routes:
        return state

    # 2-opt intra
    for r_idx, route in enumerate(routes):
        if len(route) < 3:
            continue
        for i in range(len(route) - 1):
            for j in range(i + 2, len(route)):
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                new_routes = list(routes)
                new_routes[r_idx] = new_route
                new_seq = []
                for r in new_routes:
                    new_seq.extend(r)
                    new_seq.append(0)
                new_seq.pop()
                ms = calc_makespan_fn(new_seq, env_data)
                if ms < best_ms:
                    best_ms = ms
                    routes[r_idx] = new_route
                    seq = new_seq

    # relocate inter (with safe index tracking)
    if len(routes) > 1:
        best_improvement = True
        while best_improvement:
            best_improvement = False
            for r1 in range(len(routes)):
                if not routes[r1]:
                    continue
                for i in range(len(routes[r1])):
                    cust = routes[r1][i]
                    for r2 in range(len(routes)):
                        if r1 == r2:
                            continue
                        if not routes[r2]:
                            continue
                        load2 = sum(demands[x] for x in routes[r2])
                        if load2 + demands[cust] <= cap:
                            for j in range(len(routes[r2]) + 1):
                                new_r1 = routes[r1][:i] + routes[r1][i + 1 :]
                                new_r2 = routes[r2][:j] + [cust] + routes[r2][j:]
                                new_routes = list(routes)
                                new_routes[r1] = new_r1
                                new_routes[r2] = new_r2
                                # Filter empty routes
                                new_routes_filtered = [r for r in new_routes if r]
                                new_seq = []
                                for r in new_routes_filtered:
                                    new_seq.extend(r)
                                    new_seq.append(0)
                                if new_seq:
                                    new_seq.pop()
                                ms = calc_makespan_fn(new_seq, env_data)
                                if ms < best_ms:
                                    best_ms = ms
                                    seq = new_seq
                                    routes = new_routes_filtered
                                    best_improvement = True
                                    break
                            if best_improvement:
                                break
                    if best_improvement:
                        break
                if best_improvement:
                    break

    state.sequence = seq
    state.makespan = best_ms
    return state
