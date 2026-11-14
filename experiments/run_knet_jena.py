import yaml
import argparse
import numpy as np
import tensorflow as tf
import os

from data.pipeline import TimeSeriesPipeline
from models.gru_baseline import GRUBaseline
from training.trainer import Trainer

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
    parser.add_argument("--max_points", type=int, default=20000)
    parser.add_argument("--max_knet_steps", type=int, default=2000)
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

    # Jena has no explicit state-space A,H, so use identity approximation
    config["dataset"]["A"] = 1.0
    config["dataset"]["H"] = 1.0

    # ---------------- SEED ----------------
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # ---------------- PIPELINE ----------------
    pipeline = TimeSeriesPipeline(config)
    data, _, y_norm_full = pipeline.run_ssl()

    # ---------------- LIMIT REAL-WORLD SEQUENCE ----------------
    # Using a middle chunk instead of the first chunk, to avoid only early dataset behavior.
    total_len = len(y_norm_full)
    max_points = min(args.max_points, total_len)

    start = max(0, (total_len - max_points) // 2)
    end = start + max_points

    y_norm_full = y_norm_full[start:end]

    print(f"Using Jena segment: start={start}, end={end}, length={len(y_norm_full)}")

    # Rebuild SSL windows from the limited sequence.
    W = config["window"]["input_length"]
    X_all, Y_all = build_full_ssl_windows(y_norm_full, W)

    X_all = X_all.astype(np.float32)
    Y_all = Y_all.astype(np.float32)

    # Split manually after limiting sequence.
    n = len(X_all)
    train_end = int(n * config["split"]["train"])
    val_end = int(n * (config["split"]["train"] + config["split"]["val"]))

    X_train, Y_train = X_all[:train_end], Y_all[:train_end]
    X_val, Y_val = X_all[train_end:val_end], Y_all[train_end:val_end]
    X_test, Y_test = X_all[val_end:], Y_all[val_end:]

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print("Train windows:", X_train.shape)
    print("Val windows:", X_val.shape)
    print("Test windows:", X_test.shape)

    # =========================================================
    # GRU ENCODER (SSL)
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
    # FULL RESIDUALS ON LIMITED SEQUENCE
    # =========================================================
    print("\nRunning batched inference for limited full sequence...\n")

    batch_size = 256
    preds_list = []

    X_full = X_all[..., np.newaxis]

    for i in range(0, len(X_full), batch_size):
        batch = X_full[i:i + batch_size]
        preds = model(batch, training=False)

        if isinstance(preds, tuple):
            preds = preds[0]

        preds_list.append(preds.numpy())

    y_hat_full = np.concatenate(preds_list, axis=0).squeeze()
    y_true_full = Y_all.squeeze()

    residual_full = y_true_full - y_hat_full

    print("Residual mean:", np.mean(residual_full))
    print("Residual var :", np.var(residual_full))

    # =========================================================
    # DELAY + NOISE ESTIMATION
    # =========================================================
    d_res, _ = estimate_delay(y_true_full, residual_full, max_delay=50)
    d_pred, _ = estimate_delay(y_true_full, y_hat_full, max_delay=50)

    # For real-world Jena, prediction-based delay is usually smoother/stabler.
    d_hat = d_pred

    sigma2_hat = estimate_noise_variance(residual_full)

    print(f"Input delay argument: {args.delay}")
    print(f"Estimated delay from residual: {d_res}")
    print(f"Estimated delay from prediction: {d_pred}")
    print(f"Estimated delay used: {d_hat}")
    print(f"Estimated noise variance: {sigma2_hat}")

    # =========================================================
    # ALIGN SIGNAL
    # =========================================================
    y_aligned = apply_delay_compensation(y_norm_full, d_hat)

    y_input = y_aligned.astype(np.float32)[..., np.newaxis]
    y_input = y_input[np.newaxis, :]  # (1, T, 1)

    # Limit KalmanNet sequence length to avoid OOM.
    max_knet_steps = min(args.max_knet_steps, y_input.shape[1])
    y_input_knet = y_input[:, :max_knet_steps, :]
    y_aligned_knet = y_aligned[:max_knet_steps]

    y_aligned_tf = tf.convert_to_tensor(y_aligned_knet, dtype=tf.float32)

    print("KalmanNet input shape:", y_input_knet.shape)

    # =========================================================
    # KALMANNET
    # =========================================================
    print("\nTraining KalmanNet...\n")

    knet = KalmanNetWrapper(config, return_prior=True)
    optimizer = tf.keras.optimizers.Adam(5e-4)

    knet_losses = []

    for epoch in range(10):
        with tf.GradientTape() as tape:
            x_hat_full, y_prior_full = knet(
                y_input_knet,
                sigma2_hat,
                training=True
            )

            y_prior_full = tf.squeeze(y_prior_full, axis=0)
            y_prior_full = tf.squeeze(y_prior_full, axis=-1)

            loss = tf.reduce_mean(tf.square(y_aligned_tf - y_prior_full))

        grads = tape.gradient(loss, knet.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]
        optimizer.apply_gradients(zip(grads, knet.trainable_variables))

        knet_losses.append(float(loss.numpy()))
        print(f"Epoch {epoch + 1} | KNet Loss: {loss.numpy():.4f}")

    # =========================================================
    # FINAL OUTPUT
    # =========================================================
    x_hat_full, _ = knet(y_input_knet, sigma2_hat, training=False)
    x_hat_full = x_hat_full.numpy().squeeze()

    compare_len = min(len(x_hat_full), len(y_aligned_knet))
    pred_test = x_hat_full[-compare_len:]
    true_test = y_aligned_knet[-compare_len:]

    test_rmse = np.sqrt(np.mean((pred_test - true_test) ** 2))
    test_mae = np.mean(np.abs(pred_test - true_test))

    print("\nFINAL TEST RMSE:", test_rmse)
    print("FINAL TEST MAE :", test_mae)

    # =========================================================
    # SAVE + PLOTS
    # =========================================================
    exp_name = f"sdn_knet_jena_d{args.delay}_s{args.seed}"

    saver = ResultSaver(base_dir="results/jena/knet/data")

    saver.save_predictions(
        x_true=true_test,
        x_hat=pred_test,
        y_observed=true_test,
        name=exp_name
    )

    saver.save_metrics({
        "rmse": float(test_rmse),
        "mae": float(test_mae),
        "delay": int(args.delay),
        "seed": int(args.seed),
        "model": "sdn_kalmannet",
        "estimated_delay_residual": int(d_res),
        "estimated_delay_prediction": int(d_pred),
        "estimated_delay": int(d_hat),
        "estimated_noise": float(sigma2_hat),
        "max_points": int(max_points),
        "max_knet_steps": int(max_knet_steps),
        "segment_start": int(start),
        "segment_end": int(end)
    }, exp_name)

    saver.save_history({"knet_loss": knet_losses}, exp_name)

    plotter = Plotter("results/figures/jena_climate/knet")

    plotter.plot_predictions(true_test, pred_test, name=exp_name)

    plotter.plot_full_comparison(
        true_test,
        true_test,
        pred_test,
        name=exp_name,
        jitter=True
    )

    plotter.plot_training(
        {"knet_loss": knet_losses},
        name=exp_name + "_knet"
    )


if __name__ == "__main__":
    main()