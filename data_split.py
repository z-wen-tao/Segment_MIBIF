import numpy as np
from sklearn.model_selection import train_test_split
# 数据集处理以及划分

data1 = np.load(f'data_mfcc_new/data_2016/features.npy')
label1 = np.load(f'data_mfcc_new/data_2016/label.npy')
file1 = np.load(f'data_mfcc_new/data_2016/file.npy')
segment_times1 = np.load(f'data_mfcc_new/data_2016/segment_times.npy')  # 加载片段索引

data2 = np.load(f'data_mfcc_new/data_2022/features.npy')
label2 = np.load(f'data_mfcc_new/data_2022/label.npy')
file2 = np.load(f'data_mfcc_new/data_2022/file.npy')
segment_times2 = np.load(f'data_mfcc_new/data_2022/segment_times.npy')  # 加载片段索引

data = np.concatenate((data1, data2), axis=0)
label = np.concatenate((label1, label2), axis=0)
file = np.concatenate((file1, file2), axis=0)
segment_times = np.concatenate((segment_times1, segment_times2), axis=0)  # 合并片段索引

# 将数据集按1:9的比例分成测试集和训练集，同时拆分片段索引
data_train, data_test, label_train, label_test, file_train, file_test, segment_times_train, segment_times_test = train_test_split(
    data, label, file, segment_times, test_size=0.1, random_state=42, shuffle=True
)

# 保存拆分后的数据、文件名和片段索引
np.save(f'data_mfcc_new/train_data.npy', data_train)
np.save(f'data_mfcc_new/label_train.npy', label_train)
np.save(f'data_mfcc_new/file_train.npy', file_train)
np.save(f'data_mfcc_new/segment_times_train.npy', segment_times_train)

np.save(f'data_mfcc_new/test_data.npy', data_test)
np.save(f'data_mfcc_new/label_test.npy', label_test)
np.save(f'data_mfcc_new/file_test.npy', file_test)
np.save(f'data_mfcc_new/segment_times_test.npy', segment_times_test)



