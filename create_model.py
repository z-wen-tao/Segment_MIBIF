import tensorflow as tf
from keras import layers, models
# BIGRU时序分布增强模型
class RecurrentNetworkModel(tf.keras.Model):

    def __init__(self, k):
        super(RecurrentNetworkModel, self).__init__()
        self.k = k  # 将k作为类的属性
        self.rnn = layers.Bidirectional(layers.GRU(
            256, return_sequences=True, dropout=0.1, input_shape=(747, self.k)), merge_mode='concat')
        self.linear = models.Sequential([
            layers.TimeDistributed(layers.Dense(256, activation='tanh',)),
            layers.TimeDistributed(layers.Dropout(0.1)),
            layers.TimeDistributed(layers.Dense(128, activation='tanh')),
            layers.TimeDistributed(layers.Dropout(0.1)),
            layers.TimeDistributed(layers.Dense(5))
        ])

    def call(self, inputs, training=False):

        x = self.rnn(inputs, training=training)
        print(f"RNN output shape: {x.shape}")
        x = self.linear(x, training=training)
        print(f"Linear output shape: {x.shape}")
        return x  # back to [B, C, T]
