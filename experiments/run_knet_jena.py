import yaml
import argparse
import numpy as np
import tensorflow as tf
import os

from data.pipeline import TimeSeriesPipeline
from models.gru_baseline import GRUBaseline
from training.trainer import Trainer
from training.metrics import rmse, mae

from utils.delay_estimation import estimate_delay, apply_delay_compensation
from utils.noise_estimation import estimate_noise_variance
from models.kalmannet_wrapper import KalmanNetWrapper

from analysis.saver import ResultSaver
from analysis.plots import Plotter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_full_ssl_windows(y_norm_full, window_size):
    X, Y = [], []
    for i in range(len(y_norm_full) - window_size):
        X.append(y_norm_full[i:i + window_size])
        Y.append(y_norm_full[i + window_size])
    return np.array(X), np.array(Y)


def main():
    args = parse_args()

    # ---------------- CONFIG ----------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["corruption"]["delay"] = args.delay

    # ---------------- SEED ----------------
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # ---------------- PIPELINE ----------------
    pipeline = TimeSeriesPipeline(config)

    # SSL pipeline (full sequence access)
    data, _, y_norm_full = pipeline.run_ssl()

    X_train, Y_train = data["train"]
    X_val, Y_val = data["val"]
    X_test, Y_test = data["test"]

    # cast
    X_train = X_train.astype(np.float32)[..., np.newaxis]
    X_val   = X_val.astype(np.float32)[..., np.newaxis]
    X_test  = X_test.astype(np.float32)[..., np.newaxis]

    Y_train = Y_train.astype(np.float32)
    Y_val   = Y_val.astype(np.float32)
    Y_test  = Y_test.astype(np.float32)

    # =========================================================
    # 1️⃣ GRU ENCODER (SSL)
    # =========================================================
    print("\nTraining GRU encoder...\n")

    model = GRUBaseline(
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        output_dim=1
    )

    trainer = Trainer(model, config)
    trainer.train((X_train, Y_train), (X_val, Y_val))

    # =========================================================
    # 2️⃣ RESIDUALS
    # =========================================================
    W = config["window"]["input_length"]

    X_full, Y_full = build_full_ssl_windows(y_norm_full, W)
    X_full = X_full.astype(np.float32)[..., np.newaxis]
    Y_full = Y_full.astype(np.float32)

    y_hat_full = model(X_full, training=False).numpy().squeeze()
    y_true_full = Y_full.squeeze()

    residual_full = y_true_full - y_hat_full

    print("Residual mean:", np.mean(residual_full))
    print("Residual var :", np.var(residual_full))

    # =========================================================
    # 3️⃣ DELAY + NOISE ESTIMATION
    # =========================================================
    d_hat, corrs = estimate_delay(
        y_true_full,
        residual_full,
        max_delay=50
    )

    sigma2_hat = estimate_noise_variance(residual_full)

    print(f"Estimated delay: {d_hat}")
    print(f"Estimated noise variance: {sigma2_hat}")

    # =========================================================
    # 4️⃣ ALIGN SIGNAL
    # =========================================================
    y_aligned = apply_delay_compensation(y_norm_full, d_hat)

    y_input = y_aligned.astype(np.float32)[..., np.newaxis]
    y_input = y_input[np.newaxis, ...]  # (1, T, 1)

    y_aligned_tf = tf.convert_to_tensor(y_aligned, dtype=tf.float32)

    # =========================================================
    # 5️⃣ KALMANNET
    # =========================================================
    print("\nTraining KalmanNet...\n")

    knet = KalmanNetWrapper(config)
    optimizer = tf.keras.optimizers.Adam(5e-4)

    for epoch in range(10):
        with tf.GradientTape() as tape:
            x_hat_full, y_prior_full = knet(y_input, sigma2_hat, training=True)

            y_prior_full = tf.squeeze(y_prior_full, axis=0)
            y_prior_full = tf.squeeze(y_prior_full, axis=-1)

            loss = tf.reduce_mean((y_aligned_tf - y_prior_full) ** 2)

        grads = tape.gradient(loss, knet.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]
        optimizer.apply_gradients(zip(grads, knet.trainable_variables))

        print(f"Epoch {epoch+1} | KNet Loss: {loss.numpy():.4f}")

    # =========================================================
    # 6️⃣ FINAL OUTPUT (your original evaluation style kept)
    # =========================================================
    x_hat_full, _ = knet(y_input, sigma2_hat, training=False)
    x_hat_full = x_hat_full.numpy().squeeze()

    pred_test = x_hat_full[-len(Y_test):]
    true_test = y_aligned[-len(Y_test):]

    test_rmse = np.sqrt(np.mean((pred_test - true_test) ** 2))
    test_mae  = np.mean(np.abs(pred_test - true_test))

    print("\nFINAL TEST RMSE:", test_rmse)
    print("FINAL TEST MAE :", test_mae)

    # =========================================================
    # SAVE + PLOTS
    # =========================================================
    exp_name = f"sdn_knet_jena_d{args.delay}_s{args.seed}"

    saver = ResultSaver(base_dir="results/jena/knet/data")

    # save predictions
    saver.save_predictions(
        x_true=true_test,
        x_hat=pred_test,
        y_observed=y_aligned[-len(pred_test):],
        name=exp_name
    )

    # save metrics
    saver.save_metrics({
        "rmse": float(test_rmse),
        "mae": float(test_mae),
        "delay": args.delay,
        "seed": args.seed,
        "model": "sdn_kalmannet",
        "estimated_delay": int(d_hat),
        "estimated_noise": float(sigma2_hat)
    }, exp_name)

    # plots
    plotter = Plotter()
    plot_dir = "results/mg_knet/plots"
    os.makedirs(plot_dir, exist_ok=True)

    plotter.plot_predictions(true_test, pred_test, name=f"{plot_dir}/{exp_name}")

    plotter.plot_full_comparison(
        y_norm_full[-len(pred_test):],
        y_aligned[-len(pred_test):],
        pred_test,
        name=f"{plot_dir}/{exp_name}",
        jitter=True
    )


if __name__ == "__main__":
    main()