import random


def run(env_data, state, calc_makespan_fn):
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

    splittable = [idx for idx, route in enumerate(routes) if len(route) >= 2]
    if not splittable:
        return state

    route_idx = random.choice(splittable)
    split_at = random.randint(1, len(routes[route_idx]) - 1)
    route = routes[route_idx]
    candidate_routes = (
        routes[:route_idx] + [route[:split_at], route[split_at:]] + routes[route_idx + 1 :]
    )
    candidate = []
    for candidate_route in candidate_routes:
        if candidate:
            candidate.append(0)
        candidate.extend(candidate_route)

    score = calc_makespan_fn(candidate, env_data)
    if score < 999999999.0:
        state.sequence = candidate
        state.makespan = score
    return state
