import yaml
import argparse
import numpy as np
import tensorflow as tf
import os

from data.pipeline import TimeSeriesPipeline
from models.gru_baseline import GRUBaseline
from models.kalmannet_wrapper import KalmanNetWrapper
from training.trainer import Trainer
from utils.delay_estimation import estimate_delay, apply_delay_compensation
from utils.noise_estimation import estimate_noise_variance
from analysis.saver import ResultSaver
from analysis.plots import Plotter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
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

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if config["dataset"]["name"] != "mackey_glass":
        raise ValueError("run_knet_mg.py expects dataset.name='mackey_glass'")

    config["corruption"]["delay"] = args.delay
    config["corruption"]["noise_std"] = args.noise

    # MG has no explicit A,H, so use identity approximation
    config["dataset"]["A"] = 1.0
    config["dataset"]["H"] = 1.0

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    exp_name = f"sdn_knet_mg_d{args.delay}_n{args.noise}_s{args.seed}"

    saver = ResultSaver(base_dir="results/mg_sdn_knet/data")
    plotter = Plotter()

    # ---------------- PIPELINE SSL ----------------
    pipeline = TimeSeriesPipeline(config)
    data, _, y_norm_full = pipeline.run_ssl()

    X_train, Y_train = data["train"]
    X_val, Y_val = data["val"]
    X_test, Y_test = data["test"]

    X_train = X_train.astype(np.float32)[..., np.newaxis]
    Y_train = Y_train.astype(np.float32)

    X_val = X_val.astype(np.float32)[..., np.newaxis]
    Y_val = Y_val.astype(np.float32)

    X_test = X_test.astype(np.float32)[..., np.newaxis]
    Y_test = Y_test.astype(np.float32)

    # ---------------- SSL GRU ENCODER ----------------
    print("\nTraining SSL GRU encoder...\n")

    encoder = GRUBaseline(
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        output_dim=1
    )

    trainer = Trainer(encoder, config)
    trainer.train((X_train, Y_train), (X_val, Y_val))

    # ---------------- FULL RESIDUAL ----------------
    W = config["window"]["input_length"]

    X_full, Y_full = build_full_ssl_windows(y_norm_full, W)
    X_full = X_full.astype(np.float32)[..., np.newaxis]
    Y_full = Y_full.astype(np.float32)

    # batched inference to avoid memory spike
    batch_size = 1024
    preds_list = []

    for i in range(0, len(X_full), batch_size):
        batch = X_full[i:i + batch_size]
        preds = encoder(batch, training=False)
        if isinstance(preds, tuple):
            preds = preds[0]
        preds_list.append(preds.numpy())

    y_hat_full = np.concatenate(preds_list, axis=0).squeeze()
    y_true_full = Y_full.squeeze()
    residual_full = y_true_full - y_hat_full

    print("Residual mean:", np.mean(residual_full))
    print("Residual variance:", np.var(residual_full))

    # ---------------- DELAY / NOISE ESTIMATION ----------------
    d_res, _ = estimate_delay(y_true_full, residual_full, max_delay=50)
    d_pred, _ = estimate_delay(y_true_full, y_hat_full, max_delay=50)

    # prediction-based delay is usually more stable for smooth MG
    d_hat = d_pred

    sigma2_hat = estimate_noise_variance(residual_full)

    print(f"TRUE delay: {args.delay}")
    print(f"Delay from residual: {d_res}")
    print(f"Delay from prediction: {d_pred}")
    print(f"Estimated delay used: {d_hat}")
    print(f"TRUE noise: {args.noise}")
    print(f"Estimated noise variance: {sigma2_hat}")

    # ---------------- DELAY COMPENSATION ----------------
    y_aligned = apply_delay_compensation(y_norm_full, d_hat)

    y_input = y_aligned.astype(np.float32)[..., np.newaxis]
    y_input = y_input[np.newaxis, ...]

    # memory-safe KalmanNet training
    max_knet_steps = min(2000, y_input.shape[1])
    y_input = y_input[:, :max_knet_steps, :]
    y_aligned = y_aligned[:max_knet_steps]

    y_aligned_tf = tf.convert_to_tensor(y_aligned, dtype=tf.float32)

    # ---------------- KALMANNET SSL TRAINING ----------------
    print("\nTraining KalmanNet with SSL observation loss...\n")

    knet = KalmanNetWrapper(config, return_prior=True)
    optimizer = tf.keras.optimizers.Adam(5e-4)

    knet_losses = []

    for epoch in range(10):
        with tf.GradientTape() as tape:
            x_hat_seq, y_prior_seq = knet(
                y_input,
                sigma2_hat,
                training=True
            )

            y_prior_seq = tf.squeeze(y_prior_seq, axis=0)
            y_prior_seq = tf.squeeze(y_prior_seq, axis=-1)

            loss = tf.reduce_mean(tf.square(y_aligned_tf - y_prior_seq))

        grads = tape.gradient(loss, knet.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]
        optimizer.apply_gradients(zip(grads, knet.trainable_variables))

        knet_losses.append(float(loss.numpy()))
        print(f"Epoch {epoch+1} | KNet Loss: {loss.numpy():.4f}")

    # ---------------- FINAL INFERENCE ----------------
    x_hat_full, _ = knet(y_input, sigma2_hat, training=False)
    x_hat_full = x_hat_full.numpy().squeeze()

    compare_len = min(len(x_hat_full), len(y_aligned))
    x_hat = x_hat_full[-compare_len:]
    y_eval = y_aligned[-compare_len:]

    test_rmse = np.sqrt(np.mean((x_hat - y_eval) ** 2))
    test_mae = np.mean(np.abs(x_hat - y_eval))

    print("\nSDN-KNet MG RMSE vs aligned observation:", test_rmse)
    print("SDN-KNet MG MAE  vs aligned observation:", test_mae)

    # ---------------- SAVE ----------------
    saver.save_predictions(
        x_true=y_eval,
        x_hat=x_hat,
        y_observed=y_eval,
        name=exp_name
    )

    metrics = {
        "rmse_obs": float(test_rmse),
        "mae_obs": float(test_mae),
        "true_delay": int(args.delay),
        "delay_residual": int(d_res),
        "delay_prediction": int(d_pred),
        "estimated_delay": int(d_hat),
        "true_noise": float(args.noise),
        "estimated_noise": float(sigma2_hat),
        "seed": int(args.seed),
        "model": "sdn_knet_mg"
    }

    saver.save_metrics(metrics, exp_name)
    saver.save_history(
        {
            "encoder_history": trainer.history,
            "knet_loss": knet_losses
        },
        exp_name
    )

    # ---------------- PLOTS ----------------
    plotter = Plotter("results/mg_sdn_knet/plots")

    plotter.plot_predictions(y_eval, x_hat, name=exp_name)

    plotter.plot_full_comparison(
        y_eval,
        y_eval,
        x_hat,
        name=exp_name,
        jitter=True
    )

    plotter.plot_training(
        {"knet_loss": knet_losses},
        name=exp_name + "_knet"
    )


if __name__ == "__main__":
    main()