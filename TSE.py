from matplotlib import pyplot as plt

import mel_features
from process_label import tsv_label
import librosa
import numpy as np, os
from en import get_envelope_features
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pandas as pd
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
kk = 0
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
        kk += 1
        if kk < 3:
            continue
        # if kk > 5:
        #     break
        print(kk)
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


            selector_env = SelectKBest(mutual_info_classif, k=160)
            selected_env = selector_env.fit_transform(envelopes_split, tmp_label)

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

            # 计算所有特征的互信息分数
            mi_scores = mutual_info_classif(envelopes_split, tmp_label)
            mi_scores_sorted = np.sort(mi_scores)[::-1]  # 降序排列

            # 绘制互信息分数分布
            plt.figure(figsize=(12, 6))
            sns.lineplot(x=range(1, len(mi_scores_sorted) + 1), y=mi_scores_sorted)
            plt.axvline(x=160, color='r', linestyle='--', label='Selected Features (160)')
            plt.title('Mutual Information Scores of All Features')
            plt.xlabel('Feature Rank')
            plt.ylabel('Mutual Information Score')
            plt.legend()
            plt.show()

            # 或绘制前 20 特征的条形图
            top_k = 128
            plt.figure(figsize=(12, 6))
            sns.barplot(x=list(range(1, top_k + 1)), y=mi_scores_sorted[:top_k], palette='viridis')
            plt.title(f'Top {top_k} Features by Mutual Information')
            plt.xlabel('Feature Rank')
            plt.ylabel('Mutual Information Score')
            plt.show()

            selector2 = SelectKBest(mutual_info_classif, k=24)
            selector2.fit(mfcc, tmp_label)
            # 选择特征
            mfcc= selector2.transform(mfcc)
            # Z-score 标准化 0均值 除标准差
            mean_mfcc = np.mean(mfcc, axis=0)
            std_mfcc = np.std(mfcc, axis=0)
            mfcc_normalized = (mfcc - mean_mfcc) / std_mfcc

            # # 合并特征
            # combined_features = np.concatenate((mfcc_normalized, selected_env), axis=1)  # 747 * (24+160) = 747 * 184

            # 将每个样本的特征和标签分别添加到列表中
            features.append(selected_env)  # List of arrays with shape (747, 184)
            labels.append(tmp_label)
            #
            # 2. 标准化特征
            scaler = StandardScaler()
            selected_features_scaled = scaler.fit_transform(mfcc)

            # 3. 应用 t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, learning_rate='auto')
            tsne_results = tsne.fit_transform(selected_features_scaled)

            # 4. 创建 DataFrame 存储 t-SNE 结果和标签

            tsne_df = pd.DataFrame({
                'TSNE1': tsne_results[:, 0],
                'TSNE2': tsne_results[:, 1],
                'Label': tmp_label
            })

            # 5. 绘制散点图
            plt.figure(figsize=(12, 8))
            sns.scatterplot(
                x='TSNE1', y='TSNE2',
                hue='Label',
                palette='tab10',  # 适合多类别的调色板
                data=tsne_df,
                legend='full',
                alpha=0.7
            )
            # 修改图例标签
            handles, labels = plt.gca().get_legend_handles_labels()
            # 替换图例标签
            label_map = {1: 'S1', 2: 'Sys', 3: 'S2', 4: 'Dias'}
            new_labels = [label_map[int(label)] for label in labels]
            plt.legend(handles, new_labels)
            plt.title('')
            plt.xlabel('')
            plt.ylabel('')
            plt.xticks([])
            plt.yticks([])
            plt.show()
# 将所有特征和标签转换为数组
features = np.vstack(features)  # (总样本数, 184)
labels = np.concatenate(labels)  # (总样本数,)

# 确保特征和标签的长度一致
assert features.shape[0] == labels.shape[0], "特征和标签的样本数量不一致！"

# 标准化所有特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 应用 t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, learning_rate='auto')
tsne_results = tsne.fit_transform(features_scaled)

# 创建 DataFrame 存储 t-SNE 结果和标签
tsne_df = pd.DataFrame({
    'TSNE1': tsne_results[:, 0],
    'TSNE2': tsne_results[:, 1],
    'Label': labels
})

# 绘制散点图
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='TSNE1', y='TSNE2',
    hue='Label',
    palette='tab10',  # 适合多类别的调色板
    data=tsne_df,
    legend='full',
    alpha=0.7
)

# 修改图例标签
handles, labels = plt.gca().get_legend_handles_labels()
# 替换图例标签
label_map = {0: 'Unknown', 1: 'S1', 2: 'Sys', 3: 'S2', 4: 'Dias'}
new_labels = [label_map[int(label)] for label in labels]
plt.legend(handles, new_labels)
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.xticks([])
plt.yticks([])
plt.show()