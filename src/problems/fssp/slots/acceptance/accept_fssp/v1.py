import numpy as np


def run(env_data: dict, state, calc_makespan_fn):
    # 模拟退火接收准则
    prev_mk = state.metadata.get("prev_makespan", state.makespan)
    current_mk = state.makespan

    # 初始化温度 (如果不存在)
    if "temperature" not in state.metadata:
        state.metadata["temperature"] = 100.0

    temp = state.metadata["temperature"]

    # 如果变差了，以一定概率接受
    if current_mk > prev_mk:
        delta = current_mk - prev_mk
        acceptance_prob = np.exp(-delta / max(temp, 1e-5))

        if np.random.random() > acceptance_prob:
            # 拒绝，回滚
            state.sequence = state.metadata["prev_sequence"].copy()
            state.makespan = prev_mk

    # 降温
    state.metadata["temperature"] = max(temp * 0.99, 0.1)

    return state
