import random


def run(env_data, state, calc_makespan_fn):
    seq = state.sequence[:]
    demands = env_data["demands"]
    cap = env_data["capacity"]
    n = env_data["num_customers"]

    custs = [x for x in seq if x != 0]
    num_ruin = min(n, max(3, n // 6))
    removed = set(random.sample(custs, num_ruin))

    routes = []
    curr = []
    for x in seq:
        if x == 0:
            if curr:
                routes.append(curr)
            curr = []
        elif x not in removed:
            curr.append(x)
    if curr:
        routes.append(curr)

    removed = list(removed)
    random.shuffle(removed)

    for c in removed:
        best_cost = float("inf")
        best_pos = None
        for r_idx in range(len(routes)):
            load = sum(demands[x] for x in routes[r_idx])
            if load + demands[c] <= cap:
                for i in range(len(routes[r_idx]) + 1):
                    new_r = routes[r_idx][:i] + [c] + routes[r_idx][i:]
                    test_routes = routes[:r_idx] + [new_r] + routes[r_idx + 1 :]
                    new_seq = []
                    for r in test_routes:
                        new_seq.extend(r)
                        new_seq.append(0)
                    new_seq.pop()
                    ms = calc_makespan_fn(new_seq, env_data)
                    if ms < best_cost:
                        best_cost = ms
                        best_pos = (r_idx, i)
        if best_pos:
            r_idx, i = best_pos
            routes[r_idx].insert(i, c)
        else:
            routes.append([c])

    new_seq = []
    for r in routes:
        new_seq.extend(r)
        new_seq.append(0)
    if new_seq:
        new_seq.pop()

    ms = calc_makespan_fn(new_seq, env_data)
    if ms < 999999999:
        state.sequence = new_seq
        state.makespan = ms
    return state
