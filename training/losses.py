import tensorflow as tf


class MSELoss:
    def __call__(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))