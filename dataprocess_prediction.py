import mel_features
import librosa
import numpy as np
import os
from en import get_envelope_features
import joblib  # 使用 joblib 保存和加载选择器

# 去除信号尖峰，基于移动窗口的方法，通过计算窗口内的最大幅值来确定是否存在尖峰，并将尖峰位置附近的样本置为较小的值
def spike_removal(signal, fs):
    window_size = round(fs / 2)
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


win_len = 0.050
hop_len = 0.02
sr = 4000
n_fft = 2048
n_mfcc = 128
t = 15
x = 2

features = []
segment_times = []
# 加载已保存的特征选择器
selector1 = joblib.load(f'selectors/selector_160.pkl')
selector2 = joblib.load(f'selectors/selector_24.pkl')

# 测试数据位于以下路径
test_data_folder = f'signals/0{x}.wav'
wav_files = [test_data_folder]

for wav_file in wav_files:
    audio, sr = librosa.load(wav_file, sr=4000)
    target = sr * t
    if len(audio) < target:
        pad_size = target - len(audio)
        audio = np.pad(audio, (0, pad_size), mode='constant')
    despiked_signal = spike_removal(audio, sr)
    mean_value = np.mean(despiked_signal)
    centered_data = despiked_signal - mean_value
    scal_max = np.max(np.abs(centered_data))
    centered_data = centered_data / scal_max
    n = len(audio)/ sr
    k_segments = int(np.ceil(n / t))
    for k_i in range(k_segments):
        # 计算音频片段的起始和结束索引
        l_audio = k_i * target
        r_audio = (k_i + 1) * target
        segment = audio[l_audio:r_audio]
        segment_start_time = k_i * target
        segment_end_time = (k_i + 1) * target
        # 检查段的长度是否不足 15 秒，如果不足，则补零
        segment_length = len(segment)
        if segment_length < t * sr:
            # 补齐零
            segment = np.pad(segment, (0, t * sr - segment_length), 'constant',
                             constant_values=0)

        # 保存片段的起始和结束时间
        segment_times.append((segment_start_time, segment_end_time))
        tmp_audio = segment

        # 提取包络特征
        envelopes_split = get_envelope_features(tmp_audio, sr)

        # 使用已保存的选择器进行特征转换
        selected_features = selector1.transform(envelopes_split)

        # 提取 MFCC 特征
        mfcc = mel_features.log_mel_spectrogram(
            tmp_audio,
            audio_sample_rate=sr,
            log_offset=0,
            window_length_secs=win_len,
            hop_length_secs=hop_len,
            num_mel_bins=n_mfcc,
            lower_edge_hertz=10,
            upper_edge_hertz=2000,
        )

        mfcc = np.nan_to_num(mfcc, nan=0.0, posinf=0.0, neginf=0.0)
        mfcc = selector2.transform(mfcc)

        # Z-score 标准化
        mean_mfcc = np.mean(mfcc, axis=0)
        std_mfcc = np.std(mfcc, axis=0)
        mfcc_normalized = (mfcc - mean_mfcc) / std_mfcc

        # 组合特征
        data1 = np.concatenate((mfcc_normalized, selected_features), axis=1)
        features.append(data1)

# 将列表转换为数组
features = np.array(features)
segment_times = np.array(segment_times)

# 创建保存目录（如果不存在）
os.makedirs(f'/0{x}', exist_ok=True)

# 保存特征和文件名
np.save(f'0{x}/features.npy', features)
np.save(f'0{x}/segment_times.npy', segment_times)



