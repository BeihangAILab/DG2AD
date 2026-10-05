def run(env_data, state, calc_makespan_fn):
    dist = env_data["dist_matrix"]
    demands = env_data["demands"]
    capacity = env_data["capacity"]
    depot = env_data["depot_idx"]

    routes, current = [], []
    for node in state.sequence:
        if node == 0:
            if current:
                routes.append(current)
            current = []
        else:
            current.append(node)
    if current:
        routes.append(current)
    if len(routes) < 2:
        return state

    loads = [sum(demands[node] for node in route) for route in routes]
    best_delta = 0.0
    best_swap = None

    for first_idx in range(len(routes) - 1):
        first = routes[first_idx]
        for second_idx in range(first_idx + 1, len(routes)):
            second = routes[second_idx]
            for first_pos, first_node in enumerate(first):
                first_prev = first[first_pos - 1] if first_pos > 0 else depot
                first_next = first[first_pos + 1] if first_pos + 1 < len(first) else depot
                for second_pos, second_node in enumerate(second):
                    new_first_load = loads[first_idx] - demands[first_node] + demands[second_node]
                    new_second_load = loads[second_idx] - demands[second_node] + demands[first_node]
                    if new_first_load > capacity + 1e-9 or new_second_load > capacity + 1e-9:
                        continue

                    second_prev = second[second_pos - 1] if second_pos > 0 else depot
                    second_next = second[second_pos + 1] if second_pos + 1 < len(second) else depot
                    delta = (
                        dist[first_prev, second_node]
                        + dist[second_node, first_next]
                        - dist[first_prev, first_node]
                        - dist[first_node, first_next]
                        + dist[second_prev, first_node]
                        + dist[first_node, second_next]
                        - dist[second_prev, second_node]
                        - dist[second_node, second_next]
                    )
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_swap = (first_idx, first_pos, second_idx, second_pos)

    if best_swap is None:
        return state

    first_idx, first_pos, second_idx, second_pos = best_swap
    candidate_routes = [route[:] for route in routes]
    candidate_routes[first_idx][first_pos], candidate_routes[second_idx][second_pos] = (
        candidate_routes[second_idx][second_pos],
        candidate_routes[first_idx][first_pos],
    )
    candidate = []
    for route in candidate_routes:
        if candidate:
            candidate.append(0)
        candidate.extend(route)

    score = calc_makespan_fn(candidate, env_data)
    if score < state.makespan - 1e-9:
        state.sequence = candidate
        state.makespan = score
    return state
