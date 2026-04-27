import tensorflow as tf

class PaperGRUCell(tf.keras.layers.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def build(self, input_shape):
        input_dim = input_shape[-1]

        initializer = tf.keras.initializers.GlorotUniform()

        self.W_r = self.add_weight(
            shape=(input_dim, self.hidden_size),
            initializer=initializer,
            trainable=True,
            name="W_r"
        )
        self.U_r = self.add_weight(
            shape=(self.hidden_size, self.hidden_size),
            initializer=initializer,
            trainable=True,
            name="U_r"
        )

        self.W_z = self.add_weight(
            shape=(input_dim, self.hidden_size),
            initializer=initializer,
            trainable=True,
            name="W_z"
        )
        self.U_z = self.add_weight(
            shape=(self.hidden_size, self.hidden_size),
            initializer=initializer,
            trainable=True,
            name="U_z"
        )

        self.W = self.add_weight(
            shape=(input_dim, self.hidden_size),
            initializer=initializer,
            trainable=True,
            name="W"
        )
        self.U = self.add_weight(
            shape=(self.hidden_size, self.hidden_size),
            initializer=initializer,
            trainable=True,
            name="U"
        )

        self.b_r = self.add_weight(
            shape=(self.hidden_size,),
            initializer="zeros",
            trainable=True,
            name="b_r"
        )
        self.b_z = self.add_weight(
            shape=(self.hidden_size,),
            initializer="zeros",
            trainable=True,
            name="b_z"
        )
        self.b_h = self.add_weight(
            shape=(self.hidden_size,),
            initializer="zeros",
            trainable=True,
            name="b_h"
        )

    def call(self, x_t, h_prev):
        r_t = tf.sigmoid(
            tf.matmul(x_t, self.W_r) +
            tf.matmul(h_prev, self.U_r) +
            self.b_r
        )

        z_t = tf.sigmoid(
            tf.matmul(x_t, self.W_z) +
            tf.matmul(h_prev, self.U_z) +
            self.b_z
        )

        h_tilde = tf.tanh(
            tf.matmul(x_t, self.W) +
            tf.matmul(r_t * h_prev, self.U) +
            self.b_h
        )

        h_t = z_t * h_prev + (1.0 - z_t) * h_tilde
        return h_t
    
class GRUEncoder(tf.keras.Model):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = PaperGRUCell(hidden_size)

    def call(self, x):
        # x shape: (batch, time_steps, features)

        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]

        h = tf.zeros((batch_size, self.hidden_size))

        for t in range(time_steps):
            x_t = x[:, t, :]
            h = self.cell(x_t, h)

        # final hidden state = summary vector c
        c = h
        return c
    
class ChoGRUEncoderPredictor(tf.keras.Model):
    def __init__(self, hidden_size=64, output_dim=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = PaperGRUCell(hidden_size)
        self.output_layer = tf.keras.layers.Dense(output_dim)

    def call(self, x):
        # x shape: (batch, time_steps, features)

        batch_size = tf.shape(x)[0]
        time_steps = x.shape[1]

        h = tf.zeros((batch_size, self.hidden_size))

        for t in range(time_steps):
            x_t = x[:, t, :]
            h = self.cell(x_t, h)

        # final hidden state = encoder summary vector c
        c = h

        # prediction head
        y_hat = self.output_layer(c)

        return y_hat