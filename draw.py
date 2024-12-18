# 状态名称和颜色，包括状态4
import librosa
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import medfilt
import os

# 设置字体和参数，解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决坐标轴负号显示为方块的问题
matplotlib.rcParams['font.size'] = 14

state_names = ['S1', 'sys', 'S2', 'dia']
state_colors = ['green', 'red', 'pink', 'lightblue']


# 平滑处理函数（中值滤波）
def smooth_states(states, kernel_size=5):
    return medfilt(states, kernel_size=kernel_size)

def plot_heart_sound_segmentation(states, datas, state_names, state_colors):
    """
    绘制心音信号及其对应的状态标签。

    参数：
    - states (list of np.ndarray): 每个记录的状态标签数组。
    - datas (list of np.ndarray): 每个记录的心音信号数组。
    - state_names (list of str): 状态名称列表。
    - state_colors (list of str): 状态对应的颜色列表。
    """
    num_records = len(datas)
    fig, axes = plt.subplots(num_records, 1, figsize=(15, 5 * num_records), sharex=True)

    # 如果只有一个记录，axes 不是列表，转换为列表
    if num_records == 1:
        axes = [axes]

    for idx in range(num_records):
        ax = axes[idx]
        record_data = datas[idx]
        record_state = states[idx]

        # 如果需要平滑处理，可以取消注释
        # record_state = smooth_states(record_state)  # 应用平滑处理

        num_samples = len(record_data)
        sample_rate = 4000  # 采样率4000Hz，确保与数据一致
        time = np.arange(num_samples) / sample_rate

        ax.plot(time, record_data, label='Heart Sound')
        frame_step = 0.02  # 帧步长0.02秒

        # 找到状态变化的索引
        state_labels = record_state
        state_changes = np.where(np.diff(state_labels) != 0)[0] + 1
        start_indices = np.concatenate(([0], state_changes))
        end_indices = np.concatenate((state_changes, [len(state_labels)]))

        for start_idx, end_idx in zip(start_indices, end_indices):
            state_label = state_labels[start_idx]
            start_time = start_idx * frame_step
            end_time = end_idx * frame_step

            # 获取颜色和标签
            color = state_colors[state_label]
            label = state_names[state_label]

            # 绘制连续状态的区域
            ax.axvspan(start_time, end_time, color=color, alpha=0.6, label=label)

        ax.set_title(f'Record{idx + 1}')
        ax.set_xlabel('Time(s)')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')
        # 设置图例，避免重复
        handles, labels_ = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()

k = 160
state_path = f'C:/workbanch/segment_MIBF/new_4'
state_paths = f'{state_path}/state.npy'

# 加载测试集数据、标签、文件名和片段索引
path1 = 'C:/workbanch/heat_sound/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0/training/'
path2 = 'C:/workbanch/heat_sound/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data'
file_test = np.load(f'C:/workbanch/segment_MIBF/data_mfcc_new/file_test.npy')
state = np.load(state_paths)
segment_times_test = np.load(f'C:/workbanch/segment_MIBF/data_mfcc_new/segment_times_test.npy')
t = 15

for i, j, s in zip(file_test, segment_times_test, state):
    print(j)
    print(i)
    base = os.path.basename(i)
    # 尝试路径1
    p_path = os.path.join(path1, base)
    # 如果路径1的文件不存在，则使用路径2
    if not os.path.exists(p_path):
        p_path = os.path.join(path2, base)

    audio, sr = librosa.load(p_path, sr=4000)
    target_sample = int(sr * t)
    if len(audio) < target_sample:
        pad_size = target_sample - len(audio)
        audio = np.pad(audio, (0, pad_size), mode='constant')
    x = j[0]
    y = j[1]
    tmp_audio = audio[x:y]
    plot_heart_sound_segmentation([s], [tmp_audio], state_names, state_colors)

