import tensorflow as tf
import mel_features
from process_label import tsv_label
import librosa
import numpy as np, os
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from en import get_envelope_features
# 处理2016数据集，提取特征

win_len = 0.050
hop_len = 0.02
n_mfcc = 128
t = 15
features = list()
labels = list()
signals = list()
file = list()
segment_times = []

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
list1 = ['a', 'b', 'c', 'd', 'e', 'f'] # e中存在未标注的心音记录
tsv_file = []
for i in list1:

    # 文件夹路径
    data_f = 'D:/desktop/2016/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0'
    data_folder = os.path.join(data_f, f'training-{i}')

    # 获取文件夹中所有.wav文件的路径
    wav_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.wav')]
    # tsv文件目录
    tsv_f = 'D:/desktop/2016//classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0/annotations/end_time'  # 对原始数据集数据添加一列——开始时间
    tsv_folder = os.path.join(tsv_f, f'training-{i}_StateAns')

    for wav_file in wav_files:
        audio, sr = librosa.load(wav_file, sr=4000)

        audio_length_sec = len(audio) / sr
        signals.append(audio)

        target = sr * t
        if len(audio) < target:
            pad_size = target - len(audio)
            audio = np.pad(audio, (0, pad_size), mode='constant')
        despiked_signal = spike_removal(audio, sr)  #
        mean_value = np.mean(despiked_signal)
        centered_data = despiked_signal - mean_value
        scal_max = np.max(np.abs(centered_data))
        centered_data = centered_data / scal_max

        n = len(audio)
        k = n // target

        file_name = os.path.basename(wav_file)
        if i == 'e':
            file_name = file_name[:6] + "_StateAns.tsv"
        else:
            file_name = file_name[:5] + "_StateAns.tsv"
        tsv_now = os.path.join(tsv_folder, file_name)
        tsv_file.append(tsv_now)
        label = tsv_label.process_tsv(tsv_now, audio_length_sec)

        fs_features = int(1 / hop_len)

        for k_i in range(0, k):
            # 计算音频片段的起始和结束索引
            l_audio = k_i * target
            r_audio = (k_i + 1) * target

            tmp_audio = centered_data[l_audio:r_audio]

            # 计算片段的起始和结束时间（秒）
            segment_start_time = l_audio / sr
            segment_end_time = r_audio / sr

            # 计算标签的起始和结束索引
            label_start_idx = int(segment_start_time * fs_features)
            label_end_idx = int(segment_end_time * fs_features)

            # 确保索引不超过标签的长度
            label_start_idx = min(label_start_idx, len(label))
            label_end_idx = min(label_end_idx, len(label))

            # 提取对应的标签
            tmp_label = label[label_start_idx:label_end_idx]

            # 确保标签长度为 747
            desired_length = 747
            current_length = len(tmp_label)

            if current_length > desired_length:
                # 截断标签
                tmp_label = tmp_label[:desired_length]
            elif current_length < desired_length:
                # 填充标签
                padding = np.full((desired_length - current_length,), 0, dtype=int)
                tmp_label = np.concatenate((tmp_label, padding), axis=0)

            # 提取包络特征
            envelopes_split = get_envelope_features(tmp_audio, sr)
            selector = SelectKBest(mutual_info_classif, k=160)
            selector.fit(envelopes_split, tmp_label)
            # 选择特征
            selected_features = selector.transform(envelopes_split)
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
            selector2 = SelectKBest(mutual_info_classif, k=24)
            selector2.fit(mfcc, tmp_label)
            # 选择特征
            mfcc = selector2.transform(mfcc)
            # Z-score 标准化 0均值 除标准差
            mean_mfcc = np.mean(mfcc, axis=0)
            std_mfcc = np.std(mfcc, axis=0)
            mfcc_normalized = (mfcc - mean_mfcc) / std_mfcc

            data1 = np.concatenate((mfcc_normalized, selected_features), axis=1)
            count_label = list(tmp_label)
            if count_label.count(0) >= 600:
                continue
            segment_start_time = k_i * target
            segment_end_time = (k_i + 1) * target
            # 保存片段的起始和结束时间
            segment_times.append((segment_start_time, segment_end_time))
            labels.append(tmp_label)
            features.append(data1)
            file.append(wav_file)

features = np.array(features)
labels = tf.one_hot(labels, 5)
labels = np.array(labels)
file = np.array(file)

segment_times = np.array(segment_times)

# 保存训练和测试特征以及标签
np.save(f'data_mfcc_new/data_2016/label', labels)
np.save(f'data_mfcc_new/data_2016/features', features)
np.save(f'data_mfcc_new/data_2016/file', file)
np.save(f'data_mfcc_new/data_2016/segment_times.npy', segment_times)


