import numpy as np
import scipy.stats as sci_statim
import pyximport; pyximport.install()
import v_hmm
# HSMM
def get_systolic_interval(posterior, fs, heart_rate, min_duration=150):
    heart_cycle_samples = (60 / heart_rate) * fs
    max_systolic_duration = int(
        heart_cycle_samples / 2
    )
    min_systolic_duration = round(min_duration * 1e-3 * fs)
    mhs_posterior = np.sum(posterior[:, [0, 2]], axis=1)
    mhs_acf = np.correlate(mhs_posterior, mhs_posterior, mode="full")
    mhs_acf = mhs_acf[len(mhs_acf) // 2:]
    mhs_acf = mhs_acf / mhs_acf[0]
    valid_acf = mhs_acf[min_systolic_duration: max_systolic_duration + 1]
    try:
        peak = np.argmax(valid_acf)
        absolute_peak = min_systolic_duration + peak
        return absolute_peak / fs
    except ValueError:
        print("Attempt to get argmax of empty systolic interval..")
        print(
            "HR:{}, min/max:{}/{}".format(heart_rate, min_systolic_duration, max_systolic_duration)
        )
        raise ValueError("Systolic interval function failed.")

def get_duration_distributions(heart_rate, systolic_interval, fs):
    distrib_S1 = sci_statim.norm(loc=0.1163 * fs, scale=0.0196 * fs)
    distrib_S2 = sci_statim.norm(loc=0.1032 * fs, scale=0.0195 * fs)
    mean_sys = (systolic_interval * fs) - (0.1279 * fs)
    mean_sys = max(mean_sys, 0.07 * fs)
    std_sys = 0.025 * fs
    mean_dia = (((60 / heart_rate) - systolic_interval) * fs) - (0.1053 * fs)
    mean_dia = max(mean_dia, 0.1 * fs)
    std_diastole = 0.050 * fs
    distrib_sys = sci_statim.norm(loc=mean_sys, scale=std_sys)
    distrib_dia = sci_statim.norm(loc=mean_dia, scale=std_diastole)

    return distrib_S1, distrib_sys, distrib_S2, distrib_dia

def get_heart_rate(posterior, fs, min_val=30, max_val=150, states=[1]):
    systolic_posterior = np.sum(posterior[:, states], axis=1)

    acf = np.correlate(systolic_posterior, systolic_posterior, mode="full")
    acf = acf[len(acf) // 2:]
    acf = acf / acf[0]

    min_index = round((60 / max_val) * fs)
    max_index = round((60 / min_val) * fs)

    valid_acf = acf[min_index: max_index + 1]

    rel_peak_loc = np.argmax(valid_acf)
    absolute_peak_loc = min_index + rel_peak_loc
    heart_cycle_time = absolute_peak_loc / fs

    return 60 / heart_cycle_time


def get_duration_matrix(d, distributions):
    duration_vectors = [distrib.pdf(d) for distrib in distributions]
    duration_matrix = np.stack(duration_vectors).T
    duration_matrix = duration_matrix / np.sum(
        duration_matrix, axis=0
    )
    return duration_matrix


def double_duration_viterbi(posteriors, fs):

    min_hr = 30
    max_hr = 180
    min_systole = 150
    max_duration = 1
    hr_states = [0, 1, 2, 4]

    heart_rate = get_heart_rate(posteriors, fs, min_hr, max_hr, hr_states)  # 心率
    systolic_interval = get_systolic_interval(posteriors, fs, heart_rate, min_systole)  # 收缩期间隔
    duration_distributions = get_duration_distributions(heart_rate, systolic_interval, fs)

    max_duration = int((60 / heart_rate) * fs * max_duration)
    d = np.arange(1, max_duration + 1)
    duration_matrix = get_duration_matrix(d, duration_distributions)
    transition_matrix = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    z_posteriors = np.zeros((posteriors.shape[0], 4), dtype=posteriors.dtype)
    z_posteriors[:, 0] = posteriors[:, 1]
    z_posteriors[:, 1] = posteriors[:, 2]
    z_posteriors[:, 2] = posteriors[:, 3]
    z_posteriors[:, 3] = posteriors[:, 4]
    states = v_hmm.hsmm_viterbi(
        z_posteriors,
        duration_matrix.astype(np.float32),
        max_duration,
        transition_matrix.astype(np.float32),
    )
    return states