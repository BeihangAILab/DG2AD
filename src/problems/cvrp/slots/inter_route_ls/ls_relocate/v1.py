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
    best_move = None

    for src_idx, source in enumerate(routes):
        for src_pos, node in enumerate(source):
            prev_node = source[src_pos - 1] if src_pos > 0 else depot
            next_node = source[src_pos + 1] if src_pos + 1 < len(source) else depot
            removal_delta = (
                dist[prev_node, next_node] - dist[prev_node, node] - dist[node, next_node]
            )
            for dst_idx, target in enumerate(routes):
                if dst_idx == src_idx:
                    continue
                if loads[dst_idx] + demands[node] > capacity + 1e-9:
                    continue
                for dst_pos in range(len(target) + 1):
                    left = target[dst_pos - 1] if dst_pos > 0 else depot
                    right = target[dst_pos] if dst_pos < len(target) else depot
                    insertion_delta = dist[left, node] + dist[node, right] - dist[left, right]
                    delta = removal_delta + insertion_delta
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_move = (src_idx, src_pos, dst_idx, dst_pos)

    if best_move is None:
        return state

    src_idx, src_pos, dst_idx, dst_pos = best_move
    candidate_routes = [route[:] for route in routes]
    node = candidate_routes[src_idx].pop(src_pos)
    candidate_routes[dst_idx].insert(dst_pos, node)
    candidate_routes = [route for route in candidate_routes if route]
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
