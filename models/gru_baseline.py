import tensorflow as tf


class GRUBaseline(tf.keras.Model):
    def __init__(self, hidden_size=64, num_layers=1, output_dim=1):
        super(GRUBaseline, self).__init__()

        self.gru_layers = []

        for i in range(num_layers):
            return_sequences = (i < num_layers - 1)
            self.gru_layers.append(
                tf.keras.layers.GRU(
                    hidden_size,
                    return_sequences=return_sequences
                )
            )

        self.dense = tf.keras.layers.Dense(output_dim)

    def call(self, x):
        # x shape: (batch, time_steps, features)

        for gru in self.gru_layers:
            x = gru(x)

        output = self.dense(x)
        return output