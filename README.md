# Segment_MIBIF
Heart Sound Segmentation

训练模型代码：

2016_mfcc 2022_mfcc 用于处理数据以及提取特征

数据集一：https://physionet.org/content/challenge-2016/1.0.0/ [1]

数据集二：https://physionet.org/content/challenge-2022/1.0.0/ [2]

data_split用于划分数据集

create_model搭建BIGRU模型

train_kfold用于训练模型

load_model用于推理以及搭建HSMM模型，并预测心音分割结果

draw用于绘制测试集心音分割结果

测试算法代码：

selectors已保存的特征选择器

datapocess_prediction用于提取测试数据的特征

dataprocess_hsmm对测试数据进行模型推理以及hsmm，预测心音分割结果

dataprocess_draw绘制测试数据的心音分割结果

signals提供了一些可供测试的数据，也可仔细导入，更改其中datapocess_prediction的test_data_folder路径即可

需要的环境保存在requirements.txt
