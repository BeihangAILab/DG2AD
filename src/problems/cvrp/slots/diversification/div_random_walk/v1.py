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
    routes = _routes(original)
    demands = env_data["demands"]
    capacity = float(env_data["capacity"])
    changed = False

    for _ in range(min(5, max(1, int(env_data["num_customers"])))):
        nonempty = [index for index, route in enumerate(routes) if route]
        if not nonempty:
            break
        source_index = random.choice(nonempty)
        source = routes[source_index]
        customer_index = random.randrange(len(source))
        customer = source[customer_index]
        target_index = random.randrange(len(routes))
        target = routes[target_index]
        if source_index == target_index and len(source) < 2:
            continue
        if source_index != target_index:
            target_load = sum(float(demands[value]) for value in target)
            if target_load + float(demands[customer]) > capacity + 1e-9:
                continue
        source.pop(customer_index)
        if not source and source_index != target_index:
            routes.pop(source_index)
            if source_index < target_index:
                target_index -= 1
            target = routes[target_index]
        insertion = random.randrange(len(target) + 1)
        target.insert(insertion, customer)
        changed = True

    candidate = _sequence(routes)
    if changed and candidate != original:
        score = calc_makespan_fn(candidate, env_data)
        if score < 999999999:
            state.sequence = candidate
            state.makespan = score
    return state
