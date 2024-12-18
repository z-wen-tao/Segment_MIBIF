import pandas as pd
import numpy as np

# 采样率和总帧数
sampling_rate = 4000  # Hz
WINDOW_STEP = 0.02
fs_features = int(1 / WINDOW_STEP)
win = 0.05
step = 0.02

def process_tsv(tsv_data, audio_length_sec):
    filename = tsv_data
    segmentation_df = pd.read_csv(filename, sep="\t", header=None)
    segmentation_df.columns = ["start", "end", "state"]
    # 计算总帧数
    total_frames = int(audio_length_sec * fs_features)
    # 创建一个全0的标签数组
    segmentation_label = np.zeros(total_frames, dtype=int)
    # 对每一帧进行处理
    for frame_idx in range(total_frames):
        frame_start = frame_idx * step
        frame_end = frame_start + win
        # 查找所有与当前帧重叠的分割区间
        overlapping_segments = segmentation_df[
            (segmentation_df['start'] < frame_end) & (segmentation_df['end'] > frame_start)]
        # 如果有多个重叠的分段，计算每个段占比
        if len(overlapping_segments) > 1:
            max_overlap = 0
            max_state = 0
            for _, segment in overlapping_segments.iterrows():
                overlap_start = max(frame_start, segment['start'])
                overlap_end = min(frame_end, segment['end'])
                overlap_duration = max(0, overlap_end - overlap_start)
                # 计算重叠占比
                overlap_ratio = overlap_duration / win
                # 选择占比最大的状态
                if overlap_ratio > max_overlap:
                    max_overlap = overlap_ratio
                    max_state = segment['state']
            segmentation_label[frame_idx] = max_state
        elif len(overlapping_segments) == 1:
            # 如果只有一个重叠分段，直接使用该段状态
            segmentation_label[frame_idx] = overlapping_segments['state'].values[0]
    return segmentation_label




