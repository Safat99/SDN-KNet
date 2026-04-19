import tensorflow as tf


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))