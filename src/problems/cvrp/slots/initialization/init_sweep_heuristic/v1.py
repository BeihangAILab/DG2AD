def run(env_data, state, calc_makespan_fn):
    dist = env_data["dist_matrix"]
    demands = env_data["demands"]
    capacity = env_data["capacity"]
    n = env_data["num_customers"]
    depot = env_data["depot_idx"]

    # Compute pseudo-angles based on distance proxy if coordinates are not available.
    # Since only dist_matrix is provided, we simulate sweep by clustering nodes based on distance.
    unvisited = set(range(1, n + 1))
    sequence = []

    while unvisited:
        seed = min(unvisited, key=lambda x: dist[depot, x])
        route = [seed]
        unvisited.remove(seed)
        curr_load = demands[seed]

        while unvisited:
            best_dist = float("inf")
            best_node = None
            for nxt in unvisited:
                if curr_load + demands[nxt] <= capacity:
                    if dist[route[-1], nxt] < best_dist:
                        best_dist = dist[route[-1], nxt]
                        best_node = nxt
            if best_node is None:
                break
            route.append(best_node)
            curr_load += demands[best_node]
            unvisited.remove(best_node)

        sequence.extend(route)
        if unvisited:
            sequence.append(0)

    state.sequence = sequence
    state.makespan = calc_makespan_fn(sequence, env_data)
    if not hasattr(state, "metadata") or state.metadata is None:
        state.metadata = {}
    return state
