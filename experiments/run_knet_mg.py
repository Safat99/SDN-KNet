import yaml
import numpy as np
import tensorflow as tf
import argparse
import os 

from data.pipeline import TimeSeriesPipeline
from models.kalmannet_wrapper import KalmanNetWrapper
from training.trainer import Trainer

from analysis.saver import ResultSaver
from analysis.plots import Plotter
from training.metrics import rmse, mae


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def main():

    args = parse_args()

    # ---------------- LOAD CONFIG ----------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    if dataset_name != "mackey_glass":
        raise ValueError(
            f"run_knet_mg.py expects dataset.name='mackey_glass', got '{dataset_name}'"
        )

    # override corruption
    config["corruption"]["delay"] = args.delay
    config["corruption"]["noise_std"] = args.noise

    # ---------------- 🔥 FIX FOR KALMANNET ----------------
    # MG has no A, H → inject identity dynamics
    config["dataset"]["A"] = 1.0
    config["dataset"]["H"] = 1.0

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

    # reshape for GRU-style input
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # ---------------- MODEL ----------------
    model = KalmanNetWrapper(config, return_prior=False)

    # ---------------- TRAIN ----------------
    trainer = Trainer(model, config)
    trainer.train((X_train, Y_train), (X_val, Y_val))

    # ---------------- TEST ----------------
    preds = model(X_test, training=False)

    # safety (in case wrapper returns tuple in future)
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = preds.numpy().squeeze()
    Y_test = Y_test.squeeze()

    # ---------------- METRICS ----------------
    test_rmse = float(rmse(Y_test, preds).numpy())
    test_mae  = float(mae(Y_test, preds).numpy())

    metrics = {
        "rmse": test_rmse,
        "mae": test_mae,
        "delay": args.delay,
        "noise": args.noise,
        "seed": args.seed,
        "model": "kalmannet",

        # debug stats
        "n_test": len(Y_test),
        "mean_pred": float(np.mean(preds)),
        "std_pred": float(np.std(preds)),
        "mean_true": float(np.mean(Y_test)),
        "std_true": float(np.std(Y_test))
    }

    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    # ---------------- SAVE ----------------
    exp_name = f"knet_mg_d{args.delay}_n{args.noise}_s{args.seed}"

    saver = ResultSaver(base_dir="results/mg_knet/data")
    saver.save_history(trainer.history, exp_name)

    saver.save_predictions(
        x_true=Y_test,
        x_hat=preds,
        y_observed=y_norm[-len(preds):],
        name=exp_name
    )

    saver.save_metrics(metrics, exp_name)

    # ---------------- PLOTTING ----------------
    plotter = Plotter()

    plot_dir = "results/mg_knet/plots"
    os.makedirs(plot_dir, exist_ok=True)

    plotter.plot_training(trainer.history, name=f"{plot_dir}/{exp_name}")
    plotter.plot_predictions(Y_test, preds, name=f"{plot_dir}/{exp_name}")

    plotter.plot_full_comparison(
        x_norm[-len(preds):],
        y_norm[-len(preds):],
        preds,
        name=f"{plot_dir}/{exp_name}",
        jitter=True
    )


if __name__ == "__main__":
    main()