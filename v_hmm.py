import numpy as np

# 维特比算法
def hsmm_viterbi(posteriors, durations, max_duration, A):
    T, N = posteriors.shape
    psi_np = np.empty((T + max_duration, N), dtype=np.float32)
    psi_np[:] = -np.inf
    psi_np[0, :] = np.log(posteriors[0, :])  # Uniform priors so not multiplied in here
    psi_arg_np = np.empty((T + max_duration, N), dtype=np.intc)
    psi_duration_np = np.empty((T + max_duration, N), dtype=np.intc)
    delta = psi_np
    psi = psi_arg_np
    psi_duration = psi_duration_np
    for t in range(1, T + max_duration):
        for s in range(N):
            for d in range(1, max_duration + 1):
                start_t = max(0, min(t - d, T - 1))  # clamp (t-d) to [0, T-2] range
                end_t = min(t, T)
                delta_max = float("-inf")
                i_max = -1
                for i in range(N):
                    temp_delta = delta[start_t, i] + np.log(A[i][s] + np.finfo(float).eps)
                    if temp_delta > delta_max:
                        delta_max = temp_delta
                        i_max = i
                product_observation_probs = 0
                for i in range(start_t, end_t):
                    product_observation_probs += np.log(posteriors[i, s] + np.finfo(float).eps)
                delta_this_duration = delta_max + product_observation_probs + np.log(durations[d - 1, s] + np.finfo(float).eps)
                if delta_this_duration > delta[t, s]:
                    delta[t, s] = delta_this_duration
                    psi[t, s] = i_max
                    psi_duration[t, s] = d
    current_state = -1
    end_time = -1
    max_delta_after = float("-inf")
    for t in range(T, T + max_duration):
        for s in range(N):
            if delta[t, s] > max_delta_after:
                current_state = s
                end_time = t
                max_delta_after = delta[t, s]

    states_np = -np.ones(T + max_duration, dtype=np.int32)
    states = states_np
    states[end_time] = current_state
    t = end_time
    while t > 0:
        d = psi_duration[t, current_state]
        for i in range(max(0, t - d), t):
            states[i] = current_state

        t = max(0, t - d)
        current_state = psi[t, current_state]

    return states[:T]
