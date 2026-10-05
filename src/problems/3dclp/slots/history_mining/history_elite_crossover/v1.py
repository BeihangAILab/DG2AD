import random
import numpy as np


def run(env_data, state, calc_makespan_fn):
    """Ordered crossover between current and best-known priority permutations."""
    n = env_data["num_activities"]
    if n < 2:
        return state
    original = np.asarray(state.sequence, dtype=np.int32).copy()
    elite_raw = state.metadata.get("best_seq")
    elite = np.asarray(elite_raw, dtype=np.int32).ravel() if elite_raw is not None else original
    if len(elite) != n or set(elite.tolist()) != set(range(n)):
        elite = original
    state.metadata["prev_sequence"] = original.copy()
    state.metadata["prev_makespan"] = state.makespan

    left, right = sorted(random.sample(range(n), 2))
    right += 1
    child = np.full(n, -1, dtype=np.int32)
    child[left:right] = original[left:right]
    used = set(child[left:right].tolist())
    fill = [int(item) for item in elite if int(item) not in used]
    positions = list(range(right, n)) + list(range(0, left))
    for position, item in zip(positions, fill):
        child[position] = item
    if np.array_equal(child, original):
        child[0], child[1] = child[1], child[0]
    state.sequence = child
    state.makespan = calc_makespan_fn(child, env_data)
    return state
