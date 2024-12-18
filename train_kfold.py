import keras
import tensorflow as tf
import numpy as np
from keras import callbacks
from sklearn.model_selection import KFold
import create_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

batch_size = 64
lr = 0.001

tf.debugging.set_log_device_placement(True)
k = 184
data_train = np.load(f'data_mfcc_new/train_data.npy')
label_train = np.load(f'data_mfcc_new/label_train.npy')
data_test = np.load(f'data_mfcc_new/test_data.npy')
label_test = np.load(f'data_mfcc_new/label_test.npy')
print('train:', data_train.shape[0])
print('train', data_train.shape[2])


class PrintLossAndAcc(callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        print("For batch {}, loss is {:7.2f}, accuracy is {:7.2f}.".format(batch, logs["loss"], logs["accuracy"]))

print_loss_acc = PrintLossAndAcc()
model = create_model.RecurrentNetworkModel(k)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
for train, test in kfold.split(data_train, label_train):

    csv_logger = callbacks.CSVLogger(f'training_{fold}.log')

    callback = callbacks.EarlyStopping(monitor='val_loss', patience=10)
    checkpoint = callbacks.ModelCheckpoint(filepath=f"new_6", monitor="val_loss", save_best_only=True,
                                           save_weights_only=False, mode="min")

    model.fit(data_train[train], label_train[train], validation_data=(data_train[test], label_train[test]), batch_size=batch_size, epochs=1000,
              callbacks=[callback, csv_logger, checkpoint, print_loss_acc])  # , reduce_lr
    fold += 1


best_model = keras.models.load_model(f"new_6")

# # 在测试集上评估最终模型性能
test_loss, test_accuracy = best_model.evaluate(data_test, label_test, batch_size=32)

print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

print(k)



