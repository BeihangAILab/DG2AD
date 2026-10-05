import math
import random


def _routes(sequence):
    routes = []
    current = []
    for value in sequence:
        if value == 0:
            if current:
                routes.append(current)
                current = []
        else:
            current.append(int(value))
    if current:
        routes.append(current)
    return routes


def _sequence(routes):
    flattened = []
    for route in routes:
        if not route:
            continue
        if flattened:
            flattened.append(0)
        flattened.extend(route)
    return flattened


def run(env_data, state, calc_makespan_fn):
    original = list(state.sequence)
    customers = [value for value in original if value != 0]
    if len(customers) < 2:
        return state

    distance = env_data["dist_matrix"]
    demands = env_data["demands"]
    capacity = float(env_data["capacity"])
    anchor = random.choice(customers)
    ruin_count = min(len(customers), max(2, int(math.ceil(0.10 * len(customers)))))
    removed = set(sorted(customers, key=lambda value: float(distance[anchor, value]))[:ruin_count])
    routes = [
        retained
        for route in _routes(original)
        if (retained := [value for value in route if value not in removed])
    ]

    for customer in sorted(removed, key=lambda value: float(distance[anchor, value])):
        best = None
        for route_index, route in enumerate(routes):
            load = sum(float(demands[value]) for value in route)
            if load + float(demands[customer]) > capacity + 1e-9:
                continue
            for position in range(len(route) + 1):
                previous = 0 if position == 0 else route[position - 1]
                following = 0 if position == len(route) else route[position]
                delta = (
                    float(distance[previous, customer])
                    + float(distance[customer, following])
                    - float(distance[previous, following])
                )
                proposal = (delta, route_index, position)
                if best is None or proposal < best:
                    best = proposal
        if best is None:
            if float(demands[customer]) > capacity + 1e-9:
                return state
            routes.append([customer])
        else:
            _, route_index, position = best
            routes[route_index].insert(position, customer)

    candidate = _sequence(routes)
    score = calc_makespan_fn(candidate, env_data)
    if candidate != original and score < 999999999:
        state.sequence = candidate
        state.makespan = score
    return state
