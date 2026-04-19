import yaml
import numpy as np
import tensorflow as tf

from data.pipeline import TimeSeriesPipeline
from models.gru_baseline import GRUBaseline
from training.trainer import Trainer


def main():
    # Load config
    with open("configs/mackey_glass.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Pipeline
    pipeline = TimeSeriesPipeline(config)
    data, _ = pipeline.run()

    X_train, Y_train = data["train"]
    X_val, Y_val = data["val"]
    X_test, Y_test = data["test"]

    # reshape for GRU: (batch, time, features)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Model
    model = GRUBaseline(
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        output_dim=config["window"]["forecast_horizon"]
    )

    # Trainer
    trainer = Trainer(model, config)

    # Train
    trainer.train(
        (X_train, Y_train),
        (X_val, Y_val)
    )

    # Test evaluation
    preds = model(X_test)
    mse = tf.reduce_mean(tf.square(Y_test - preds))
    print("Test MSE:", mse.numpy())


if __name__ == "__main__":
    main()