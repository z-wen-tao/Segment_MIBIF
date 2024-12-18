# Segment_MIBIF
Heart Sound Segmentation

如果你想要进行模型训练，训练模型代码：

第一步：运行2016_mfcc 2022_mfcc 用于处理数据以及提取特征

  数据集一：https://physionet.org/content/challenge-2016/1.0.0/ [1]
  
  数据集二：https://physionet.org/content/challenge-2022/1.0.0/ [2]

第二步：运行data_split用于划分数据集

第三步：运行train_kfold用于训练模型

第四步：运行load_model用于推理以及搭建HSMM模型，并预测心音分割结果

draw用于绘制测试集心音分割结果

create_model搭建BIGRU模型

如果你想使用算法进行心音分割，测试算法代码：

第一步：datapocess_prediction用于提取测试数据的特征，你可以把其中test_data_folder路径更改为自己的数据路径

第二步：dataprocess_hsmm对测试数据进行模型推理以及hsmm，预测心音分割结果

你可以通过运行dataprocess_draw，进而绘制测试数据的心音分割结果

signals提供了一些可供测试的数据，也可自行导入，更改其中datapocess_prediction的test_data_folder路径即可

selectors已保存的特征选择器

以上操作需要的环境保存在requirements.txt


[1] Classification of Normal/Abnormal Heart Sound Recordings: the PhysioNet/Computing in Cardiology Challenge 2016
Gari D. Clifford, Chengyu Liu, Benjamin Moody, David Springer, Ikaro Silva, Qiao Li, Roger G. Mark

[2] Reyna, M., Kiarashi, Y., Elola, A., Oliveira, J., Renna, F., Gu, A., Perez Alday, E. A., Sadr, N., Mattos, S., Coimbra, M., Sameni, R., Bahrami Rad, A., Koscova, Z., & Clifford, G. (2023). Heart Murmur Detection from Phonocardiogram Recordings: The George B. Moody PhysioNet Challenge 2022 (version 1.0.0). PhysioNet. 

