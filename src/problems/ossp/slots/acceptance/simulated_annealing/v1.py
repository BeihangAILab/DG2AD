import numpy as np
import math


def run(env_data, state, calc_makespan_fn):
    md = state.metadata
    prev_mk = md.get("prev_makespan", state.makespan)
    current_mk = state.makespan
    best_mk = md.get("best_makespan", state.makespan)
    best_seq = md.get("best_sequence", state.sequence.copy())
    temp = md.get("temp", 300.0)
    iteration = md.get("iteration", 0)
    reheat_count = md.get("reheat_count", 0)
    no_improve_global = md.get("no_improve_global", 0)
    record = md.get("record", best_mk)
    deviation = md.get("deviation", max(best_mk * 0.03, 1.5))
    elite_pool = md.get("elite_pool", [])

    if current_mk < best_mk:
        best_mk = current_mk
        best_seq = state.sequence.copy()
        no_improve_global = 0
        record = best_mk
        deviation = max(best_mk * 0.025, 1.0)
        if len(elite_pool) < 5:
            elite_pool.append((current_mk, state.sequence.copy()))
        else:
            worst_idx = max(range(len(elite_pool)), key=lambda i: elite_pool[i][0])
            if current_mk < elite_pool[worst_idx][0]:
                elite_pool[worst_idx] = (current_mk, state.sequence.copy())
    else:
        no_improve_global += 1

    accepted = True
    if current_mk > prev_mk:
        delta = current_mk - prev_mk
        if current_mk <= record + deviation:
            accepted = True
        else:
            prob = math.exp(-delta / max(temp, 0.01))
            if np.random.random() > prob:
                accepted = False

    if not accepted:
        state.sequence = md["prev_sequence"].copy()
        state.makespan = prev_mk
        md["no_improve"] = md.get("no_improve", 0) + 1
    else:
        md["no_improve"] = 0 if current_mk <= prev_mk else md.get("no_improve", 0)

    no_imp = md.get("no_improve", 0)
    phase = iteration / max(iteration + 500, 1)
    alpha = 0.998 - 0.008 * phase - (0.003 if no_imp > 30 else 0)
    alpha = max(alpha, 0.985)
    temp = max(temp * alpha, 0.02)
    deviation = max(deviation * 0.9985, 0.3)

    if no_imp > 60:
        reheat_count += 1
        md["no_improve"] = 0
        if reheat_count <= 8:
            temp = max(180.0 / (1 + reheat_count * 0.2), 12.0)
            deviation = max(best_mk * 0.025, 1.0)
            if elite_pool and np.random.random() < 0.5:
                pick = elite_pool[np.random.randint(len(elite_pool))]
                state.sequence = pick[1].copy()
                state.makespan = pick[0]
            else:
                state.sequence = best_seq.copy()
                state.makespan = best_mk
        else:
            seq = best_seq.copy()
            n = len(seq)
            seg_len = max(n // 4, 5)
            start = np.random.randint(0, max(n - seg_len, 1))
            seq[start : start + seg_len] = seq[start : start + seg_len][::-1]
            for _ in range(max(n // 6, 4)):
                i, j = np.random.randint(0, n, size=2)
                seq[[i, j]] = seq[[j, i]]
            state.sequence = seq
            state.makespan = calc_makespan_fn(state, env_data)
            if state.makespan < best_mk:
                best_mk = state.makespan
                best_seq = state.sequence.copy()
            temp = 150.0
            deviation = max(best_mk * 0.035, 1.5)
            reheat_count = 0

    if no_improve_global > 200:
        no_improve_global = 0
        temp = 250.0
        deviation = max(best_mk * 0.035, 2.0)
        reheat_count = 0
        if elite_pool:
            pick = elite_pool[np.random.randint(len(elite_pool))]
            state.sequence = pick[1].copy()
            state.makespan = pick[0]

    md["temp"] = temp
    md["best_makespan"] = best_mk
    md["best_sequence"] = best_seq
    md["iteration"] = iteration + 1
    md["reheat_count"] = reheat_count
    md["no_improve_global"] = no_improve_global
    md["record"] = record
    md["deviation"] = deviation
    md["elite_pool"] = elite_pool
    return state
