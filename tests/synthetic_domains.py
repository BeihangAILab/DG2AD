"""Small in-memory instances used by network-free domain tests."""

from __future__ import annotations

import numpy as np


def _packing(problem: str):
    sizes = np.asarray([[3, 2, 2], [2, 3, 2], [2, 2, 3], [1, 1, 1]], dtype=np.int32)
    env = {
        "name": f"synthetic_{problem}",
        "bin_size": np.asarray([6, 6, 6], dtype=np.int32),
        "num_activities": len(sizes),
        "item_sizes": sizes,
        "item_volumes": np.prod(sizes, axis=1).astype(np.int32),
        "lower_bound": 1.0,
        "upper_bound": 1.0,
    }
    if problem == "3dclp":
        env["fill_upper_bound"] = 100.0 * float(np.sum(env["item_volumes"])) / 216.0
        env["orientation_allowed"] = np.ones((len(sizes), 3), dtype=np.int8)
    return env


def _cvrp():
    coords = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    return {
        "name": "synthetic_cvrp",
        "dist_matrix": dist,
        "demands": np.asarray([0, 1, 1, 1], dtype=np.float64),
        "capacity": 2.0,
        "num_nodes": 4,
        "num_customers": 3,
        "depot_idx": 0,
        "opt_cost": 4.0,
        "lower_bound": 4.0,
        "upper_bound": 4.0,
    }


def _fjsp():
    machines = np.asarray(
        [
            [[0, 1], [0, 1]],
            [[0, 1], [0, 1]],
        ],
        dtype=np.int32,
    )
    times = np.asarray(
        [
            [[2, 4], [3, 1]],
            [[4, 2], [2, 3]],
        ],
        dtype=np.int32,
    )
    processing = np.asarray(
        [
            [[2, 4], [3, 1]],
            [[4, 2], [2, 3]],
        ],
        dtype=np.int32,
    )
    return {
        "name": "synthetic_fjsp",
        "num_jobs": 2,
        "num_machines": 2,
        "max_ops": 2,
        "total_ops": 4,
        "op_counts": np.asarray([2, 2], dtype=np.int32),
        "eligible_counts": np.full((2, 2), 2, dtype=np.int32),
        "machines_matrix": machines,
        "times_matrix": times,
        "processing_matrix": processing,
        "lower_bound": 4.0,
        "upper_bound": 8.0,
    }


def _fssp():
    return {
        "name": "synthetic_fssp",
        "num_jobs": 3,
        "num_machines": 2,
        "processing_times": np.asarray([[2, 4, 3], [3, 2, 1]], dtype=np.int32),
        "total_ops": 6,
        "lower_bound": 7.0,
        "upper_bound": 10.0,
    }


def _jsp():
    return {
        "name": "synthetic_jsp",
        "num_jobs": 2,
        "num_machines": 2,
        "total_ops": 4,
        "machines_matrix": np.asarray([[0, 1], [1, 0]], dtype=np.int32),
        "times_matrix": np.asarray([[2, 3], [1, 4]], dtype=np.int32),
        "lower_bound": 5.0,
        "upper_bound": 8.0,
    }


def _max_cut():
    return {
        "name": "synthetic_max_cut",
        "num_nodes": 3,
        "neighbors": np.asarray([1, 2, 0, 2, 0, 1], dtype=np.int32),
        "edge_weights": np.ones(6, dtype=np.float64),
        "offsets": np.asarray([0, 2, 4, 6], dtype=np.int32),
        "best_known_cut": 2.0,
        "lower_bound": -2.0,
        "upper_bound": -2.0,
    }


def _mis():
    return {
        "name": "synthetic_mis",
        "num_vertices": 6,
        "adj": [{1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4}],
        "lower_bound": -3.0,
        "upper_bound": -3.0,
    }


def _ossp():
    return {
        "name": "synthetic_ossp",
        "num_jobs": 2,
        "num_machines": 2,
        "total_ops": 4,
        "ossp_times": np.asarray([[2, 3], [4, 1]], dtype=np.int32),
        "lower_bound": 5.0,
        "upper_bound": 8.0,
    }


def _rcpsp():
    adj = np.zeros((4, 4), dtype=np.int32)
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    for source, target in edges:
        adj[source, target] = 1
    return {
        "name": "synthetic_rcpsp",
        "num_activities": 4,
        "num_resources": 1,
        "durations": np.asarray([0, 2, 3, 0], dtype=np.int32),
        "requests": np.asarray([[0], [1], [1], [0]], dtype=np.int32),
        "capacities": np.asarray([2], dtype=np.int32),
        "adj_matrix": adj,
        "edges": np.asarray(edges, dtype=np.int32),
        "lower_bound": 3.0,
        "upper_bound": 5.0,
    }


def _salbp():
    adj = np.zeros((4, 4), dtype=np.int32)
    adj[0, 2] = 1
    adj[1, 3] = 1
    return {
        "name": "synthetic_salbp",
        "num_tasks": 4,
        "cycle_time": 5,
        "times": np.asarray([2, 3, 2, 1], dtype=np.int32),
        "adj_matrix": adj,
        "lower_bound": 2.0,
        "upper_bound": 3.0,
    }


def _tsp():
    coords = np.asarray([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float64)
    return {
        "name": "synthetic_tsp",
        "num_nodes": 4,
        "coords": coords,
        "distance_matrix": np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2),
        "lower_bound": 8.0,
        "upper_bound": 8.0,
    }


def synthetic_instance(problem: str):
    factories = {
        "3dbbp": lambda: _packing("3dbbp"),
        "3dclp": lambda: _packing("3dclp"),
        "cvrp": _cvrp,
        "fjsp": _fjsp,
        "fssp": _fssp,
        "jsp": _jsp,
        "max_cut": _max_cut,
        "mis": _mis,
        "ossp": _ossp,
        "rcpsp": _rcpsp,
        "salbp": _salbp,
        "tsp": _tsp,
    }
    return factories[problem]()
