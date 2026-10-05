import numpy as np


def run(env_data: dict, state, calc_makespan_fn):
    new_seq = state.sequence.copy()
    n = len(new_seq)
    if n < 2:
        return state

    # 单点插入变异 (Shift/Insert)
    # 随机拔出一个工序，插入到另一个随机位置
    different_pairs = [
        (i, j) for i in range(n) for j in range(n) if i != j and new_seq[i] != new_seq[j]
    ]
    if not different_pairs:
        return state
    i, j = different_pairs[np.random.randint(len(different_pairs))]

    val = new_seq[i]
    new_seq = np.delete(new_seq, i)
    new_seq = np.insert(new_seq, j, val)

    if np.array_equal(new_seq, state.sequence):
        new_seq = state.sequence.copy()
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]

    # 保存历史用于回滚
    state.metadata["prev_sequence"] = state.sequence.copy()
    state.metadata["prev_makespan"] = state.makespan

    state.sequence = new_seq
    state.makespan = calc_makespan_fn(state, env_data)
    return state
