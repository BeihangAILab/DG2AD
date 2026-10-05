def run(env_data, state, calc_makespan_fn):
    dist = env_data["dist_matrix"]
    demands = env_data["demands"]
    capacity = env_data["capacity"]
    n = env_data["num_customers"]
    depot = env_data["depot_idx"]

    savings = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = dist[depot, i] + dist[depot, j] - dist[i, j]
            savings.append((s, i, j))

    savings.sort(key=lambda x: x[0], reverse=True)

    routes = [[i] for i in range(1, n + 1)]
    node_to_route = {i: i - 1 for i in range(1, n + 1)}

    for s, i, j in savings:
        r_i = node_to_route[i]
        r_j = node_to_route[j]
        if r_i != r_j:
            route_i = routes[r_i]
            route_j = routes[r_j]
            if route_i[0] == i or route_i[-1] == i:
                if route_j[0] == j or route_j[-1] == j:
                    load_i = sum(demands[x] for x in route_i)
                    load_j = sum(demands[x] for x in route_j)
                    if load_i + load_j <= capacity:
                        if route_i[0] == i:
                            route_i = route_i[::-1]
                        if route_j[-1] == j:
                            route_j = route_j[::-1]
                        new_route = route_i + route_j
                        routes[r_i] = new_route
                        for x in new_route:
                            node_to_route[x] = r_i
                        routes[r_j] = []

    routes = [r for r in routes if r]
    seq = []
    for r in routes:
        seq.extend(r)
        seq.append(0)
    if seq:
        seq.pop()

    state.sequence = seq
    state.makespan = calc_makespan_fn(seq, env_data)
    if not hasattr(state, "metadata") or state.metadata is None:
        state.metadata = {}
    return state
