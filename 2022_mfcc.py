import tensorflow as tf
import mel_features
from process_label import tsv_label2022
import librosa
import numpy as np, os
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from en import get_envelope_features

# 处理2022数据集，提取特征


win_len = 0.050
hop_len = 0.02
sr = 4000
n_mfcc = 128
t = 15
WINDOW_STEP = hop_len
fs_features = int(1 / WINDOW_STEP)
data_folder = "C:/workbanch/heat_sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data"
filenames = list()
for f in sorted(os.listdir(data_folder)):
    root, extension = os.path.splitext(f)
    if not root.startswith('.') and extension == '.txt':
        filename = os.path.join(data_folder, f)
        filenames.append(filename)

roots = [os.path.split(filename)[1][:-4] for filename in filenames]
if all(float(root).is_integer() for root in roots):
    filenames = sorted(filenames, key=lambda filename: int(os.path.split(filename)[1][:-4]))
patient_files = filenames
num_patient_files = len(patient_files)
classes = ['Present', 'Unknown', 'Absent']
outcome_classes = ['Abnormal', 'Normal']
features = list()
labels = list()
file = list()
segment_times = []
# 函数：将时间转换为帧号
def time_to_frame(time, frame_duration):
    return int(time // frame_duration)

def get_num_locations(data):
    num_locations = None
    for i, l in enumerate(data.split('\n')):
        if i==0:
            try:
                num_locations = int(l.split(' ')[1])
            except:
                pass
        else:
            break
    return num_locations

# 去除信号尖峰，基于移动窗口的方法，通过计算窗口内的最大幅值来确定是否存在尖峰，并将尖峰位置附近的样本置为较小的值
def spike_removal(signal, fs):
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

def frame(data, window_length, hop_length):

    num_samples = data.shape[0]  # 总样本数
    num_frames = int(np.floor((num_samples - window_length)/hop_length))  # 总帧数
    # print(num_frames)
    shape = (num_frames, window_length) + data.shape[1:]  # (帧数量,帧长度)
    strides = (data.strides[0] * hop_length,) + data.strides
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

for i in range(num_patient_files):
    with open(patient_files[i], 'r') as f:
        current_patient_data = f.read()
        for l in current_patient_data.split('\n'):
            if l.startswith('#Murmur:'):
                try:
                    label = l.split(': ')[1]
                except:
                    pass
        num_locations = get_num_locations(current_patient_data)

        recording_information = current_patient_data.split('\n')[1:num_locations + 1]

        recording_files = []
        recording_labels = []
        if label in ['Absent', 'Unknown']:   # 排除unknown情况,'Unknown'
            for i in range(num_locations):
                entries = recording_information[i].split(' ')
                try:
                    recording_file = entries[2]
                    recording_label = entries[3]
                except Exception as err:
                    print("error:", entries, recording_information)
                recording_files.append(recording_file)
                recording_labels.append(recording_label)
        elif label in ['Present']:
            for l in current_patient_data.split('\n'):
                if l.startswith('#Murmur locations:'):
                    try:
                        recordings = l.split(': ')[1].strip()
                        recordings = recordings.split('+')
                        for i in range(num_locations):
                            entries = recording_information[i].split(' ')
                            if entries[0] in recordings:
                                recording_file = entries[2]
                                recording_label = entries[3]
                                recording_files.append(recording_file)
                                recording_labels.append(recording_label)
                    except:
                        pass
        curr_recording_files = recording_files
        curr_labels = recording_labels
        for f, j in zip(curr_labels, curr_recording_files):
            filename = os.path.join(data_folder, j)
            audio, sr = librosa.load(filename, sr=4000)
            audio_length_sec = len(audio) / sr  # 获取音频时长（秒）
            print(audio_length_sec)
            target_sample = int(sr * t)  # t 是片段长度（秒）
            if len(audio) < target_sample:
                pad_size = target_sample - len(audio)
                audio = np.pad(audio, (0, pad_size), mode='constant')
                audio_length_sec = len(audio) / sr  # 更新音频时长
            despiked_signal = spike_removal(audio, sr)
            mean_value = np.mean(despiked_signal)
            centered_data = despiked_signal - mean_value
            scal_max = np.max(np.abs(centered_data))
            centered_data = centered_data / scal_max
            n = len(audio)
            k = n // target_sample  # 计算可以分割的片段数量
            label = tsv_label2022.process_tsv(f, audio_length_sec)
            for k_i in range(0, k):
                # 计算音频片段的起始和结束索引
                l_audio = k_i * target_sample
                r_audio = (k_i + 1) * target_sample
                tmp_audio = centered_data[l_audio:r_audio]
                segment_start_time = k_i * target_sample
                segment_end_time = (k_i + 1) * target_sample
                # 保存片段的起始和结束时间
                segment_times.append((segment_start_time, segment_end_time))
                # 计算片段的起始和结束时间（秒）
                segment_start_time = l_audio / sr
                segment_end_time = r_audio / sr

                # 计算标签的起始和结束索引
                label_start_idx = int(segment_start_time * fs_features)
                label_end_idx = int(segment_end_time * fs_features)

                # 确保索引不越界
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

                # 使用 MIFIF 进行特征选择
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
                features.append(data1)
                labels.append(tmp_label)
                file.append(j)

features = np.array(features)
labels = tf.one_hot(labels, 5)
labels = np.array(labels)
file = np.array(file)
segment_times = np.array(segment_times)

# 保存训练和测试特征以及标签
np.save(f'data_mfcc_new/data_2022/label', labels)
np.save(f'data_mfcc_new/data_2022/features', features)
np.save(f'data_mfcc_new/data_2022/file', file)
np.save(f'data_mfcc_new/data_2022/segment_times.npy', segment_times)

