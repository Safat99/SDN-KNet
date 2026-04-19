import yaml
import numpy as np
import tensorflow as tf
import argparse

from data.pipeline import TimeSeriesPipeline
from models.gru_baseline import GRUBaseline
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

    # optional future use (kept but not required)
    parser.add_argument("--model", type=str, default="gru")

    return parser.parse_args()


def main():

    args = parse_args()

    # ---------------- LOAD CONFIG ----------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override from SLURM
    config["corruption"]["delay"] = args.delay
    config["corruption"]["noise_std"] = args.noise

    # ---------------- SEED ----------------
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # ---------------- PIPELINE ----------------
    pipeline = TimeSeriesPipeline(config)
    data, _, x_norm, y_norm = pipeline.run()

    X_train, Y_train = data["train"]
    X_val, Y_val = data["val"]
    X_test, Y_test = data["test"]
    
    X_train = X_train.astype(np.float32)
    Y_train = Y_train.astype(np.float32)

    X_val = X_val.astype(np.float32)
    Y_val = Y_val.astype(np.float32)

    X_test = X_test.astype(np.float32)
    Y_test = Y_test.astype(np.float32)

    # reshape for GRU: (batch, time, features)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # ---------------- MODEL ----------------
    model = GRUBaseline(
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        output_dim=config["window"]["forecast_horizon"]
    )

    # ---------------- TRAIN ----------------
    trainer = Trainer(model, config)
    trainer.train((X_train, Y_train), (X_val, Y_val))

    # ---------------- TEST PREDICTIONS ----------------
    preds = model(X_test, training=False).numpy()

    # ensure consistent shapes
    preds = preds.squeeze()
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
        "hidden_size": config["model"]["hidden_size"],
        "num_layers": config["model"]["num_layers"],
        "epochs": config["training"]["epochs"]
    }

    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    # ---------------- SAVE ----------------
    exp_name = f"gru_mg_d{args.delay}_n{args.noise}_s{args.seed}"

    saver = ResultSaver()
    saver.save_history(trainer.history, exp_name)
    saver.save_predictions(Y_test, preds, exp_name)
    saver.save_metrics(metrics, exp_name)

    # ---------------- PLOTTING ----------------
    plotter = Plotter()

    plotter.plot_training(trainer.history, name=exp_name)
    plotter.plot_predictions(Y_test, preds, name=exp_name)

    # NOTE: assumes test set corresponds to tail of full sequence
    plotter.plot_full_comparison(
        x_norm[-len(preds):],
        y_norm[-len(preds):],
        preds,
        name=exp_name
    )


if __name__ == "__main__":
    main()