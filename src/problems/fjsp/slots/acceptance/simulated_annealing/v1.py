import math
import random
import domain_evaluator


def run(env_data, state, calc_makespan_fn):
    meta = state.metadata
    prev_ms, prev_seq = meta.get("prev_makespan", float("inf")), meta.get("prev_sequence")
    g_best_ms, g_best_seq = meta.get("g_best_ms", float("inf")), meta.get("g_best_seq")

    # 提取双重决策表示的备份
    prev_alloc = meta.get("prev_alloc")
    g_best_alloc = meta.get("g_best_alloc")

    temp, no_imp = meta.get("temp", 30.0), meta.get("no_improve", 0)

    if state.makespan < g_best_ms:
        g_best_ms = state.makespan
        g_best_seq = state.sequence[:]
        g_best_alloc = state.metadata["machine_alloc"].copy()
        no_imp = 0
    else:
        no_imp += 1

    delta = state.makespan - prev_ms
    if delta < 0 or random.random() < math.exp(-delta / max(0.1, temp)):
        # 接受新解，不做回滚
        pass
    elif prev_seq is not None and prev_alloc is not None:
        # 拒绝新解，回滚工序序列和机器分配
        state.sequence = prev_seq[:]
        state.metadata["machine_alloc"] = prev_alloc.copy()
        domain_evaluator.set_active_machine_allocation(prev_alloc)
        state.makespan = prev_ms

    temp *= 0.999
    if no_imp > 25:
        if g_best_seq is not None and g_best_alloc is not None:
            state.sequence = g_best_seq[:]
            state.metadata["machine_alloc"] = g_best_alloc.copy()
            domain_evaluator.set_active_machine_allocation(g_best_alloc)
            state.makespan = g_best_ms
        temp, no_imp, meta["force_reset"] = 30.0, 0, True

    meta.update(
        {
            "temp": temp,
            "no_improve": no_imp,
            "g_best_ms": g_best_ms,
            "g_best_seq": g_best_seq,
            "g_best_alloc": g_best_alloc,
            "prev_sequence": state.sequence[:],
            "prev_alloc": state.metadata["machine_alloc"].copy(),
            "prev_makespan": state.makespan,
        }
    )
    return state
