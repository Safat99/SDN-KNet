import tensorflow as tf


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config["training"]["learning_rate"]
        )

        self.loss_fn = tf.keras.losses.MeanSquaredError()

        # full history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_rmse": [],
            "val_rmse": [],
            "train_mae": [],
            "val_mae": []
        }

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            preds = self.model(x, training=True)
            loss = self.loss_fn(y, preds)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    @tf.function
    def val_step(self, x, y):
        preds = self.model(x, training=False)
        loss = self.loss_fn(y, preds)
        return loss

    # --- metrics ---
    @staticmethod
    def rmse(x_true, x_hat):
        return tf.sqrt(tf.reduce_mean(tf.square(x_true - x_hat), axis=None))

    @staticmethod
    def mae(x_true, x_hat):
        return tf.reduce_mean(tf.abs(x_true - x_hat), axis=None)

    def train(self, train_data, val_data):
        X_train, Y_train = train_data
        X_val, Y_val = val_data

        batch_size = self.config["training"]["batch_size"]
        epochs = self.config["training"]["epochs"]

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = train_dataset.shuffle(1000).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        val_dataset = val_dataset.batch(batch_size)

        for epoch in range(epochs):

            # --- TRAIN ---
            train_losses = []
            train_rmse_vals = []
            train_mae_vals = []

            for x_batch, y_batch in train_dataset:
                loss = self.train_step(x_batch, y_batch)
                preds = self.model(x_batch, training=False)

                train_losses.append(loss)
                train_rmse_vals.append(self.rmse(y_batch, preds))
                train_mae_vals.append(self.mae(y_batch, preds))

            # --- VALIDATION ---
            val_losses = []
            val_rmse_vals = []
            val_mae_vals = []

            for x_batch, y_batch in val_dataset:
                preds = self.model(x_batch, training=False)
                loss = self.loss_fn(y_batch, preds)

                val_losses.append(loss)
                val_rmse_vals.append(self.rmse(y_batch, preds))
                val_mae_vals.append(self.mae(y_batch, preds))

            # --- AGGREGATE ---
            train_loss = float(tf.reduce_mean(tf.stack(train_losses), axis=0).numpy())
            val_loss   = float(tf.reduce_mean(tf.stack(val_losses), axis=0).numpy())

            train_rmse = float(tf.reduce_mean(tf.stack(train_rmse_vals), axis=0).numpy())
            val_rmse   = float(tf.reduce_mean(tf.stack(val_rmse_vals), axis=0).numpy())

            train_mae  = float(tf.reduce_mean(tf.stack(train_mae_vals), axis=0).numpy())
            val_mae    = float(tf.reduce_mean(tf.stack(val_mae_vals), axis=0).numpy())

            # --- SAVE HISTORY ---
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            self.history["train_rmse"].append(train_rmse)
            self.history["val_rmse"].append(val_rmse)

            self.history["train_mae"].append(train_mae)
            self.history["val_mae"].append(val_mae)

            # --- LOG ---
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train: MSE={train_loss:.4f}, RMSE={train_rmse:.4f}, MAE={train_mae:.4f} | "
                f"Val:   MSE={val_loss:.4f}, RMSE={val_rmse:.4f}, MAE={val_mae:.4f}",
                flush=True
            )