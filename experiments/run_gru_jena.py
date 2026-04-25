import yaml
import numpy as np
import tensorflow as tf
import argparse
import os

from data.pipeline import TimeSeriesPipeline
from models.gru_baseline import GRUBaseline
from training.trainer import Trainer
from training.metrics import rmse, mae
from analysis.saver import ResultSaver
from analysis.plots import Plotter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    # ---------------- LOAD CONFIG ----------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if config["dataset"]["name"] != "jena":
        raise ValueError(
            f"Expected dataset.name='jena', got {config['dataset']['name']}"
        )

    # override delay (used by pipeline if needed)
    config["corruption"]["delay"] = args.delay

    # ---------------- SEED ----------------
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # ---------------- PIPELINE ----------------
    pipeline = TimeSeriesPipeline(config)
    data, _, x_norm, y_norm = pipeline.run()

    X_train, Y_train = data["train"]
    X_val, Y_val = data["val"]
    X_test, Y_test = data["test"]

    # cast
    X_train = X_train.astype(np.float32)
    Y_train = Y_train.astype(np.float32)

    X_val = X_val.astype(np.float32)
    Y_val = Y_val.astype(np.float32)

    X_test = X_test.astype(np.float32)
    Y_test = Y_test.astype(np.float32)

    # reshape for GRU
    X_train = X_train[..., np.newaxis]
    X_val   = X_val[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    # ---------------- MODEL ----------------
    model = GRUBaseline(
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        output_dim=1
    )

    trainer = Trainer(model, config)
    trainer.train((X_train, Y_train), (X_val, Y_val))

    preds = model(X_test).numpy()

    # ---------------- METRICS ----------------
    test_rmse = rmse(Y_test, preds).numpy()
    test_mae  = mae(Y_test, preds).numpy()

    print("RMSE:", test_rmse)
    print("MAE:", test_mae)

    # ---------------- SAVE ----------------
    exp_name = f"gru_jena_d{args.delay}_s{args.seed}"

    saver = ResultSaver(base_dir="results/jena/gru/data")
    saver.save_predictions(
        x_true=Y_test,
        x_hat=preds,
        y_observed=y_norm[-len(preds):],
        name=exp_name
    )

    saver.save_metrics({
        "rmse": float(test_rmse),
        "mae": float(test_mae),
        "delay": args.delay,
        "seed": args.seed,
        "model": "gru"
    }, exp_name)

    plotter = Plotter()
    
    plot_dir = "results/jena/gru/plots"
    os.makedirs(plot_dir, exist_ok=True)

    saver.save_history(trainer.history, exp_name)
    plotter.plot_training(trainer.history, name=f"{plot_dir}/{exp_name}")
    plotter.plot_predictions(Y_test, preds, name=f"{plot_dir}/{exp_name}")

    plotter.plot_full_comparison(
        x_norm[-len(preds):],
        y_norm[-len(preds):],
        preds.squeeze(),
        name=exp_name,
        jitter=True
    )


if __name__ == "__main__":
    main()