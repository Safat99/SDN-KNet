import tensorflow as tf

class KalmanNetWrapper(tf.keras.Model):
    def __init__(self, config):
        super().__init__()

        self.A = tf.constant(config["dataset"]["A"], dtype=tf.float32)
        self.H = tf.constant(config["dataset"]["H"], dtype=tf.float32)

        self.gru = tf.keras.layers.GRU(
            config["model"]["hidden_size"],
            return_sequences=True,
            return_state=True
        )

        self.fc = tf.keras.layers.Dense(1)

    def call(self, y_seq, sigma2_hat, training=False):

        T = tf.shape(y_seq)[1]

        x_hat_list = []
        y_prior_list = []

        # better init
        x_prev = y_seq[:, 0, :]
        h_state = None

        for t in range(T):
            y_t = y_seq[:, t, :]

            # -------- Prediction --------
            x_prior = self.A * x_prev
            y_prior = self.H * x_prior

            y_prior_list.append(y_prior)  # 🔥 IMPORTANT

            # -------- Innovation --------
            innovation = tf.clip_by_value(y_t - y_prior, -5.0, 5.0)

            # -------- Sigma feature --------
            sigma_feat = tf.math.log(sigma2_hat + 1e-6)
            sigma_feat = tf.ones_like(innovation) * sigma_feat

            # -------- GRU input --------
            gru_input = tf.concat([innovation, sigma_feat], axis=-1)
            gru_input = tf.expand_dims(gru_input, axis=1)

            if h_state is None:
                gru_out, h_state = self.gru(gru_input)
            else:
                gru_out, h_state = self.gru(gru_input, initial_state=h_state)

            # -------- Gain --------
            K_t = tf.tanh(self.fc(gru_out))
            K_t = tf.squeeze(K_t, axis=1)

            # -------- Update --------
            x_hat = x_prior + K_t * innovation

            x_hat_list.append(x_hat)
            x_prev = x_hat

            tf.debugging.check_numerics(x_hat, "NaN in x_hat")

        x_hat_seq = tf.stack(x_hat_list, axis=1)
        y_prior_seq = tf.stack(y_prior_list, axis=1)

        return x_hat_seq, y_prior_seq