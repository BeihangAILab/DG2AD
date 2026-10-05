"""Equivalent array-only FJSP simulation, optionally accelerated by Numba."""

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(**kwargs):
        return lambda function: function


@njit(cache=True)
def simulate_schedule(sequence, allocation, processing, op_counts, num_machines):
    num_jobs = len(op_counts)
    job_op_idx = np.zeros(num_jobs, dtype=np.int32)
    machine_avail = np.zeros(num_machines, dtype=np.float64)
    job_avail = np.zeros(num_jobs, dtype=np.float64)
    for pos in range(len(sequence)):
        job = int(sequence[pos])
        op = int(job_op_idx[job])
        if op >= int(op_counts[job]):
            return 999999999.0
        machine = int(allocation[job, op])
        if machine < 0 or machine >= num_machines:
            return 999999999.0
        duration = float(processing[job, op, machine])
        if duration < 0 or not np.isfinite(duration):
            return 999999999.0
        end = max(machine_avail[machine], job_avail[job]) + duration
        machine_avail[machine] = end
        job_avail[job] = end
        job_op_idx[job] += 1
    return float(np.max(machine_avail))
