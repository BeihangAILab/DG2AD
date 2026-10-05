def run(env_data, state, calc_makespan_fn):
    dist = env_data["dist_matrix"]
    demands = env_data["demands"]
    capacity = env_data["capacity"]
    num_customers = env_data["num_customers"]
    depot = env_data["depot_idx"]

    unvisited = set(range(1, num_customers + 1))
    sequence = []

    while unvisited:
        curr_load = 0
        curr_node = depot
        route = []
        while unvisited:
            best_dist = float("inf")
            best_node = None
            for nxt in unvisited:
                if curr_load + demands[nxt] <= capacity:
                    if dist[curr_node, nxt] < best_dist:
                        best_dist = dist[curr_node, nxt]
                        best_node = nxt
            if best_node is None:
                break
            route.append(best_node)
            curr_load += demands[best_node]
            unvisited.remove(best_node)
            curr_node = best_node

        sequence.extend(route)
        if unvisited:
            sequence.append(0)

    state.sequence = sequence
    state.makespan = calc_makespan_fn(sequence, env_data)
    if not hasattr(state, "metadata") or state.metadata is None:
        state.metadata = {}
    return state
