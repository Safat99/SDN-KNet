import tensorflow as tf


def rmse(x_true, x_hat):
    return tf.sqrt(tf.reduce_mean(tf.square(x_true - x_hat)))


def mae(x_true, x_hat):
    return tf.reduce_mean(tf.abs(x_true - x_hat))