import tensorflow as tf


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config["training"]["learning_rate"]
        )

        self.loss_fn = tf.keras.losses.MeanSquaredError()

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
            train_losses = []

            for x_batch, y_batch in train_dataset:
                loss = self.train_step(x_batch, y_batch)
                train_losses.append(loss)

            val_losses = []
            for x_batch, y_batch in val_dataset:
                loss = self.val_step(x_batch, y_batch)
                val_losses.append(loss)

            print(
                f"Epoch {epoch+1}: "
                f"Train Loss={tf.reduce_mean(train_losses):.4f}, "
                f"Val Loss={tf.reduce_mean(val_losses):.4f}"
            )