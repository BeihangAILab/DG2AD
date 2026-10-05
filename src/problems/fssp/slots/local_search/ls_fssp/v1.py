import numpy as np


def run(env_data: dict, state, calc_makespan_fn):
    best_seq = state.sequence.copy()
    best_mk = state.makespan
    n = len(best_seq)

    # 局部搜索：尝试 15 次随机插入，保留最好的
    for _ in range(15):
        candidate = best_seq.copy()

        # 连续两次插入，增加扰动深度
        for _ in range(2):
            i = np.random.randint(n)
            j = np.random.randint(n - 1)
            val = candidate[i]
            candidate = np.delete(candidate, i)
            candidate = np.insert(candidate, j, val)

        state.sequence = candidate
        mk = calc_makespan_fn(state, env_data)

        if mk < best_mk:
            best_mk = mk
            best_seq = candidate.copy()

    state.sequence = best_seq
    state.makespan = best_mk
    return state
