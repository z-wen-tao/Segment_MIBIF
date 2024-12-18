from scipy.signal import resample_poly, hilbert, butter, filtfilt, spectrogram
import biosppy
from scipy.signal.windows import hamming
import numpy as np
from pywt import downcoef
# 包络特征提取
win_len = 0.050
hop_len = 0.02
sr = 4000
win_length = int(round(win_len * sr))
hop_length = int(round(hop_len * sr))
n_fft = 2048
n_mfcc = 128
t = 15

def frame(data, window_length, hop_length):

    num_samples = data.shape[0]  # 总样本数
    num_frames = int(np.floor((num_samples - window_length)/hop_length))  # 总帧数
    shape = (num_frames, window_length) + data.shape[1:]  # (帧数量,帧长度)
    strides = (data.strides[0] * hop_length,) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

# 去除信号尖峰，基于移动窗口的方法，通过计算窗口内的最大幅值来确定是否存在尖峰，并将尖峰位置附近的样本置为较小的值
def spike_removal(signal,fs):
    window_size = round(fs/2)
    trailing_samples = len(signal) % window_size
    # 将 signal 分割为多个窗口，每个窗口的长度为 window_size
    if trailing_samples > 0:
        sampleframes = np.reshape(signal[:-trailing_samples],
                                  (window_size, int(len(signal[:-trailing_samples]) / window_size)))
    else:
        sampleframes = np.reshape(signal, (window_size, int(len(signal) / window_size)))
    # 计算每列绝对值的最大值
    MAAs = np.amax(abs(sampleframes), axis=0)
    while len(np.argwhere(MAAs > np.median(MAAs) * 3)) > 0:

        window_num = np.argmax(MAAs)

        if np.size(window_num) > 1:
            window_num = window_num[0]

        spike_position = np.argmax(abs(sampleframes[:, window_num]))

        if np.size(spike_position) > 1:
            spike_position = spike_position[0]

        zero_crossings = np.concatenate((abs(np.diff(np.sign(sampleframes[:, window_num]))) > 1, np.array([False])),
                                        axis=0)
        try:
            spike_start = max(
                np.concatenate((np.array([0]), np.argwhere(zero_crossings[0:spike_position + 1] == 1)[-1])))
        except:
            spike_start = 0

        zero_crossings[0:spike_position + 1] = 0

        try:
            spike_end = min(np.concatenate((np.argwhere(zero_crossings == 1)[0], np.array([window_size - 1]))))
        except:
            spike_end = window_size - 1

        sampleframes[spike_start:(spike_end + 1), window_num] = 0.0001

        MAAs = np.amax(abs(sampleframes), axis=0)
    despiked_signal = np.reshape(sampleframes, (np.size(sampleframes), 1))
    despiked_signal = np.concatenate((despiked_signal.ravel(), signal[len(despiked_signal):]), axis=0)

    return despiked_signal
def get_envelope_features(signal, fs):
    # 设计2阶 Butterworth带通滤波器
    order = 2
    passBand = np.array([25, 400])
    filtered, _, _ = biosppy.signals.tools.filter_signal(signal, 'butter', 'bandpass', order, passBand, fs)
    filtered = spike_removal(np.transpose(filtered), fs)

    # 参数设置
    fs_low = fs
    f_LPF = 8
    hop_l = int(hop_len * fs_low)
    win_l = int(win_len * fs_low)

    # 同态包络
    b, a = butter(order, 2 * f_LPF / fs, 'low')
    e1 = np.exp(filtfilt(b, a, np.log(np.abs(hilbert(filtered)))))
    e1[0] = e1[1]
    downs_e1, = biosppy.signals.tools.normalize(e1)
    downs_e1[0] = downs_e1[1]
    downs_e1_frame = frame(downs_e1, win_l, hop_l)

    # 希尔伯特包络
    e2 = np.abs(hilbert(filtered))
    downs_e2 = resample_poly(e2, 1, fs / fs_low)
    downs_e2, = biosppy.signals.tools.normalize(downs_e2)
    downs_e2[0] = downs_e2[1]
    downs_e2_frame = frame(downs_e2, win_l, hop_l)

    # 功率谱密度包络
    window_size = int(0.05 * fs)
    overlap_size = int(0.025 * fs)
    frequencies, _, Sxx = spectrogram(
        filtered,
        fs=fs,
        window=hamming(window_size),
        nperseg=window_size,
        noverlap=overlap_size,
        nfft=window_size,
        mode='psd'
    )
    low_limit_idx = np.where(frequencies >= 20)[0][0]
    high_limit_idx = np.where(frequencies <= 150)[0][-1]
    e3 = np.mean(Sxx[low_limit_idx:high_limit_idx + 1, :], axis=0)
    e3_normalized = (e3 - np.mean(e3)) / np.std(e3)
    target_length = len(downs_e1)
    e3_resampled = resample_poly(e3_normalized, up=target_length, down=len(e3_normalized))
    downs_e3_frame = frame(e3_resampled, win_l, hop_l)

    # 小波包络
    coeffs = downcoef('d', filtered, 'rbio3.9', level=3)
    coeffs = np.repeat(coeffs, 2 ** 3)
    d = (len(coeffs) - len(filtered)) / 2
    first = int(np.floor(d))
    last = int(len(coeffs) - np.ceil(d))
    coeffs = coeffs[first:last]
    e4 = abs(coeffs)
    downs_e4 = resample_poly(e4, 1, fs / fs_low)
    downs_e4,  = biosppy.signals.tools.normalize(downs_e4)
    downs_e4[0] = downs_e4[1]
    downs_e4_frame = frame(downs_e4, win_l, hop_l)

    # 将所有包络特征帧放入列表
    envelopes_split = np.hstack((downs_e1_frame, downs_e2_frame, downs_e3_frame, downs_e4_frame))
    return envelopes_split