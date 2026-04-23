import tensorflow as tf


class MSELoss:
    def __call__(self, x_true, x_hat):
        return tf.reduce_mean(tf.square(x_true - x_hat))