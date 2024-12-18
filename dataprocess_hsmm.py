import tensorflow as tf
import numpy as np
import seg_1
from keras.models import load_model
# 检查是否有可用的GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Available GPU devices: {gpus}")
else:
    print("No GPU available, running on CPU")

# 打印模型训练时使用的设备
x = 2
tf.debugging.set_log_device_placement(True)
data_path = f'0{x}/features.npy'
model_path = f'new_4'
data = np.load(data_path)

model = load_model(model_path)
print(len(data))
results = []
pres = []
for i in range(data.shape[0]):
    print(i)
    input_sample = np.expand_dims(np.float32(data[i]), axis=0)  # 添加批次维度
    prediction = model(input_sample)  # 对当前样本进行预测
    posteriors_mean = tf.reduce_mean(tf.nn.softmax(prediction, axis=-1), axis=0)
    posteriors_numpy = posteriors_mean.numpy()
    state = seg_1.double_duration_viterbi(posteriors_numpy, 50)
    pre = np.argmax(posteriors_numpy, axis=1)
    pres.append(pre)
    results.append(state)

results = np.array(results)
np.save(f'0{x}/state', results)
pres = np.array(pres)
np.save(f'0{x}/result', pres)
print(1)
