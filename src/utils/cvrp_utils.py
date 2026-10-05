"""Shared CVRP utilities: VRP parser, distance matrix, route construction, GAP evaluation."""

import os
import math
import random
import numpy as np

from src.core.configuration import problem_data_dir

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _cvrp_dir():
    private_fallback = os.path.join(PROJECT_ROOT, "src", "problems", "cvrp", "instances")
    return str(problem_data_dir("cvrp", private_fallback))


def parse_vrp(filepath):
    """Parse a .vrp file. Returns dict with name, dimension, capacity, coords, demands, depot, opt_cost, dist_matrix."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    info = {
        "name": None,
        "dimension": 0,
        "capacity": 100,
        "coords": {},
        "demands": {},
        "depot": 1,
        "opt_cost": None,
        "dist_matrix": None,
        "edge_weight_type": "EUC_2D",
    }

    section = None
    explicit_weights = []
    for line in lines:
        line = line.strip()
        if not line or line == "EOF":
            continue

        upper = line.upper()
        if upper.startswith("NAME"):
            info["name"] = line.split(":")[-1].strip()
        elif upper.startswith("COMMENT"):
            comment = line.split(":", 1)[-1].strip()
            import re

            m = re.search(r"Optimal value:\s*(\d+(?:\.\d+)?)", comment)
            if m:
                info["opt_cost"] = float(m.group(1))
        elif upper.startswith("DIMENSION"):
            info["dimension"] = int(line.split(":")[-1].strip())
        elif upper.startswith("CAPACITY"):
            info["capacity"] = int(line.split(":")[-1].strip())
        elif upper.startswith("EDGE_WEIGHT_TYPE"):
            info["edge_weight_type"] = line.split(":")[-1].strip()
        elif upper.startswith("NODE_COORD_SECTION"):
            section = "COORD"
            continue
        elif upper.startswith("EDGE_WEIGHT_SECTION"):
            section = "EDGE_WEIGHT"
            continue
        elif upper.startswith("DEMAND_SECTION"):
            section = "DEMAND"
            continue
        elif upper.startswith("DEPOT_SECTION"):
            section = "DEPOT"
            continue
        elif upper.startswith("DISPLAY_DATA") or upper.startswith("EDGE_WEIGHT_FORMAT"):
            continue

        if section == "COORD":
            parts = line.split()
            if len(parts) >= 3:
                nid = int(parts[0])
                info["coords"][nid] = (float(parts[1]), float(parts[2]))
        elif section == "EDGE_WEIGHT":
            parts = line.split()
            for p in parts:
                try:
                    explicit_weights.append(int(p))
                except ValueError:
                    pass
        elif section == "DEMAND":
            parts = line.split()
            if len(parts) >= 2:
                info["demands"][int(parts[0])] = int(parts[1])
        elif section == "DEPOT":
            if line == "-1":
                section = None
            else:
                info["depot"] = int(line.strip())

    # Build distance matrix from explicit weights if present
    if explicit_weights and not info["coords"]:
        n = info["dimension"]
        dist = np.zeros((n, n), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(i):
                if idx < len(explicit_weights):
                    w = float(explicit_weights[idx])
                    dist[i, j] = w
                    dist[j, i] = w
                    idx += 1
        info["dist_matrix"] = dist
        # Create synthetic coords from demands
        for vid, d in info["demands"].items():
            info["coords"][vid] = (float(vid), float(d))

    return info


def parse_sol(filepath):
    """Parse a .sol file. Returns (cost, routes_list)."""
    routes = []
    cost = None
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Route"):
                parts = line.split(":")
                if len(parts) >= 2:
                    nodes = [int(x) for x in parts[1].strip().split() if x.strip()]
                    if nodes:
                        routes.append(nodes)
            elif line.startswith("Cost"):
                try:
                    cost = float(line.split()[1])
                except ValueError:
                    cost = None
    return cost, routes


def load_instance_set(set_name):
    """Load all VRP+SOL instances from a set directory. Returns list of dicts."""
    set_dir = os.path.join(_cvrp_dir(), set_name)
    if not os.path.isdir(set_dir):
        return []

    vrp_files = sorted([f for f in os.listdir(set_dir) if f.endswith(".vrp")])
    instances = []
    for vf in vrp_files:
        vrp_path = os.path.join(set_dir, vf)
        sol_path = vrp_path.replace(".vrp", ".sol")
        info = parse_vrp(vrp_path)

        n = info["dimension"]
        vrp_ids = sorted(info["coords"].keys())
        id_to_idx = {vid: i for i, vid in enumerate(vrp_ids)}
        depot_idx = id_to_idx.get(info["depot"], 0)

        # Use explicit distance matrix if available, otherwise compute from coords
        if info.get("dist_matrix") is not None:
            dist = info["dist_matrix"]
        else:
            dist = np.zeros((n, n), dtype=np.float64)
            for i, vid_i in enumerate(vrp_ids):
                xi, yi = info["coords"][vid_i]
                for j, vid_j in enumerate(vrp_ids):
                    xj, yj = info["coords"][vid_j]
                    dist[i, j] = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

        # Demands array (0-indexed)
        demands = np.zeros(n, dtype=np.float64)
        for vid, d in info["demands"].items():
            if vid in id_to_idx:
                demands[id_to_idx[vid]] = d

        # Optimal cost from SOL file
        opt_cost = info.get("opt_cost")
        if opt_cost is None and os.path.exists(sol_path):
            opt_cost, _ = parse_sol(sol_path)

        instances.append(
            {
                "name": info["name"],
                "n_nodes": n,
                "capacity": info["capacity"],
                "depot_idx": depot_idx,
                "dist_matrix": dist,
                "demands": demands,
                "opt_cost": opt_cost,
            }
        )
    return instances


def route_cost(routes, dist):
    """Compute total cost of a set of routes."""
    total = 0.0
    for r in routes:
        if not r:
            continue
        prev = 0  # depot
        for node in r:
            total += dist[prev, node]
            prev = node
        total += dist[prev, 0]  # return to depot
    return total


def greedy_construct_with_heuristic(dist, demands, capacity, edge_priorities, depot=0, alpha=0.1):
    """Construct CVRP routes greedily using edge priorities as guidance.

    alpha: randomization parameter (0 = fully greedy, 1 = fully random)
    Returns list of routes (each route is list of 1-indexed customer nodes).
    """
    n = len(dist)
    customers = [i for i in range(n) if i != depot]
    unvisited = set(customers)
    routes = []

    while unvisited:
        route = []
        current = depot
        current_load = 0.0

        while True:
            feasible = []
            for c in unvisited:
                if demands[c] <= capacity - current_load:
                    feasible.append(c)

            if not feasible:
                break

            # Score = priority / distance (with randomization)
            scores = []
            for c in feasible:
                base = edge_priorities[current, c] / max(dist[current, c], 1e-6)
                scores.append(base)

            if not scores:
                break

            # Weighted random selection with alpha blending
            max_s = max(scores) if scores else 1.0
            if max_s <= 0:
                max_s = 1.0
            weights = []
            for s in scores:
                w = (1.0 - alpha) * (s / max_s) + alpha * random.random()
                weights.append(max(w, 1e-10))

            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            chosen_idx = random.choices(range(len(feasible)), weights=probs, k=1)[0]
            chosen = feasible[chosen_idx]

            route.append(chosen)
            current_load += demands[chosen]
            unvisited.discard(chosen)
            current = chosen

        if route:
            routes.append(route)

    return routes


def two_opt_single_route(route, dist, depot=0):
    """Apply 2-opt improvement to a single route."""
    if len(route) < 3:
        return route
    improved = True
    best_route = route[:]
    while improved:
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 2, len(best_route)):
                # Edge i→i+1 and j→j+1 replaced by i→j and i+1→j+1
                a = best_route[i]
                b = best_route[i + 1]
                c = best_route[j]
                d = best_route[j + 1] if j + 1 < len(best_route) else depot
                if j + 1 < len(best_route):
                    d = best_route[j + 1]
                    old_cost = dist[a, b] + dist[c, d]
                    new_cost = dist[a, c] + dist[b, d]
                    if new_cost < old_cost - 1e-9:
                        best_route[i + 1 : j + 1] = reversed(best_route[i + 1 : j + 1])
                        improved = True
                        break
                else:
                    old_cost = dist[a, b] + dist[c, depot]
                    new_cost = dist[a, c] + dist[b, depot]
                    if new_cost < old_cost - 1e-9:
                        best_route[i + 1 :] = reversed(best_route[i + 1 :])
                        improved = True
                        break
            if improved:
                break
    return best_route


def savings_construct_with_heuristic(dist, demands, capacity, edge_priorities, depot=0):
    """Clarke-Wright savings construction guided by heuristic edge priorities.

    Effective savings = base_savings * priority_factor.
    Returns list of routes.
    """
    n = len(dist)
    customers = [i for i in range(n) if i != depot]

    # Each customer starts in its own route: depot -> c -> depot
    routes = {c: [c] for c in customers}
    route_loads = {c: demands[c] for c in customers}

    # Compute effective savings for each edge, modified by heuristic priorities
    savings = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            ci = customers[i]
            cj = customers[j]
            base_savings = dist[depot, ci] + dist[depot, cj] - dist[ci, cj]
            # Heuristic modifier: average priority for this edge in both directions
            h_mod = (edge_priorities[ci, cj] + edge_priorities[cj, ci]) / 2.0
            effective_savings = base_savings * h_mod
            savings.append((effective_savings, ci, cj))

    # Sort by effective savings descending
    savings.sort(key=lambda x: x[0], reverse=True)

    for _, ci, cj in savings:
        # Check if ci and cj are in different routes (and both still exist)
        ri = None
        rj = None
        for rid, route in routes.items():
            if ci in route:
                ri = rid
            if cj in route:
                rj = rid
            if ri is not None and rj is not None:
                break

        if ri is None or rj is None or ri == rj:
            continue

        # Check capacity
        if route_loads[ri] + route_loads[rj] <= capacity:
            # Merge: check if ci is at end of its route and cj at start (or vice versa)
            route_i = routes[ri]
            route_j = routes[rj]

            if route_i[-1] == ci and route_j[0] == cj:
                routes[ri] = route_i + route_j
                route_loads[ri] += route_loads[rj]
                del routes[rj]
                del route_loads[rj]
            elif route_i[0] == ci and route_j[-1] == cj:
                routes[ri] = route_j + route_i
                route_loads[ri] += route_loads[rj]
                del routes[rj]
                del route_loads[rj]
            elif route_i[-1] == ci and route_j[-1] == cj:
                route_j.reverse()
                routes[ri] = route_i + route_j
                route_loads[ri] += route_loads[rj]
                del routes[rj]
                del route_loads[rj]
            elif route_i[0] == ci and route_j[0] == cj:
                route_i.reverse()
                routes[ri] = route_i + route_j
                route_loads[ri] += route_loads[rj]
                del routes[rj]
                del route_loads[rj]

    return list(routes.values())


def relocate_improve(routes, dist, demands, capacity, depot=0, max_iter=10):
    """Relocate customers between routes to reduce total cost."""
    improved = True
    iteration = 0
    while improved and iteration < max_iter:
        improved = False
        iteration += 1
        for ri, route_i in enumerate(routes):
            if not route_i:
                continue
            sum(demands[c] for c in route_i)
            for pi in range(len(route_i)):
                ci = route_i[pi]
                # Remove ci from route_i
                new_ri = route_i[:pi] + route_i[pi + 1 :]
                # Cost of new_ri (depot -> ... -> depot)
                cost_ri_old = route_cost([route_i], dist)
                cost_ri_new = route_cost([new_ri], dist) if new_ri else dist[depot, depot]

                # Try inserting ci into every position of every other route
                best_saving = 0.0
                best_move = None
                for rj, route_j in enumerate(routes):
                    if ri == rj:
                        continue
                    load_j = sum(demands[c] for c in route_j)
                    if load_j + demands[ci] > capacity:
                        continue
                    cost_rj_old = route_cost([route_j], dist)
                    # Try all insertion positions
                    for pj in range(len(route_j) + 1):
                        new_rj = route_j[:pj] + [ci] + route_j[pj:]
                        cost_rj_new = route_cost([new_rj], dist)
                        saving = (cost_ri_old + cost_rj_old) - (cost_ri_new + cost_rj_new)
                        if saving > best_saving:
                            best_saving = saving
                            best_move = (ri, pi, rj, pj)

                if best_saving > 1e-6:
                    ri, pi, rj, pj = best_move
                    ci = routes[ri][pi]
                    routes[ri] = routes[ri][:pi] + routes[ri][pi + 1 :]
                    routes[rj] = routes[rj][:pj] + [ci] + routes[rj][pj:]
                    improved = True
                    break
            if improved:
                break
    return [r for r in routes if r]


def evaluate_heuristic_on_instances(
    fn, instances, n_constructions=50, alpha=0.05, apply_2opt=True, train_mode=True
):
    """Evaluate a heuristic function on a list of CVRP instances.

    Uses a multi-start approach: savings construction + 2-opt + relocate improvement.
    fn: edge_priorities = fn(edge_attr, node_attr) -> np.ndarray (n×n)
    Returns (avg_gap, best_gap, results_dict).
    """
    n_starts = n_constructions if not train_mode else max(3, n_constructions // 10)
    gaps = []
    details = {}

    for inst in instances:
        dist = inst["dist_matrix"]
        demands = inst["demands"]
        capacity = inst["capacity"]
        depot = inst["depot_idx"]
        opt = inst.get("opt_cost")
        n = inst["n_nodes"]

        if opt is None:
            continue

        # Build inputs for heuristic
        edge_attr = dist.copy()
        node_attr = np.zeros((n, 2), dtype=np.float64)
        node_attr[:, 0] = demands
        node_attr[depot, 1] = 1.0

        # Get edge priorities from heuristic
        try:
            priorities = fn(edge_attr, node_attr)
            if priorities is None or not isinstance(priorities, np.ndarray):
                priorities = np.ones_like(dist)
            priorities = np.abs(priorities) + 1e-10
        except Exception:
            priorities = np.ones_like(dist)

        best_cost = float("inf")
        for _ in range(n_starts):
            # Use savings-based construction guided by heuristic
            routes = savings_construct_with_heuristic(dist, demands, capacity, priorities, depot)
            if apply_2opt:
                routes = [two_opt_single_route(r, dist, depot) for r in routes]
            routes = relocate_improve(routes, dist, demands, capacity, depot, max_iter=5)
            # Also try greedy construction with some randomization
            routes2 = greedy_construct_with_heuristic(
                dist, demands, capacity, priorities, depot, alpha
            )
            if apply_2opt:
                routes2 = [two_opt_single_route(r, dist, depot) for r in routes2]
            routes2 = relocate_improve(routes2, dist, demands, capacity, depot, max_iter=5)

            cost1 = route_cost(routes, dist)
            cost2 = route_cost(routes2, dist)
            best_cost = min(best_cost, cost1, cost2)

        if best_cost < float("inf") and opt > 0:
            gap = (best_cost - opt) / opt * 100.0
            gaps.append(gap)
            details[inst["name"]] = {"cost": best_cost, "opt": opt, "gap": gap}

    avg_gap = float(np.mean(gaps)) if gaps else 999.0
    best_gap = float(min(gaps)) if gaps else 999.0
    return avg_gap, best_gap, details
