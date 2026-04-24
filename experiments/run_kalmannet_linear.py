import yaml
import argparse
import numpy as np
import tensorflow as tf

from data.pipeline import TimeSeriesPipeline
from models.gru_baseline import GRUBaseline
from training.trainer import Trainer
from utils.delay_estimation import estimate_delay, apply_delay_compensation
from utils.noise_estimation import estimate_noise_variance
from models.kalmannet_wrapper import KalmanNetWrapper



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

    config["corruption"]["delay"] = args.delay
    config["corruption"]["noise_std"] = args.noise

    # --------------- SEED ------------------
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # ---------------- PIPELINE ----------------
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
    model = GRUBaseline(
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        output_dim=1
    )

    trainer = Trainer(model, config)
    trainer.train((X_train, Y_train), (X_val, Y_val))

    # ---------------- TEST RESIDUAL ----------------
    y_hat_test = model(X_test, training=False).numpy().squeeze()
    y_true_test = Y_test.squeeze()
    residual_test = y_true_test - y_hat_test
    
    print("Test residual mean:", np.mean(residual_test))
    print("Test residual variance:", np.var(residual_test))
    print("Test residual shape:", residual_test.shape)

    # ---------------- FULL-SEQUENCE RESIDUAL FOR DELAY/NOISE ----------------
    W = config["window"]["input_length"]

    X_full, Y_full = build_full_ssl_windows(y_norm_full, W)
    X_full = X_full.astype(np.float32)[..., np.newaxis]
    Y_full = Y_full.astype(np.float32)

    y_hat_full_ssl = model(X_full, training=False).numpy().squeeze()
    y_true_full_ssl = Y_full.squeeze()
    residual_full = y_true_full_ssl - y_hat_full_ssl

    print("Full residual mean:", np.mean(residual_full))
    print("Full residual variance:", np.var(residual_full))
    print("Full residual shape:", residual_full.shape)
    
    # ---------------- ESTIMATION ----------------
    d_hat, corrs = estimate_delay(
        y_true_full_ssl,
        residual_full,
        max_delay=50
    )
    
    sigma2_hat = estimate_noise_variance(residual_full)
    
    print(f"Estimated delay: {d_hat}")
    print(f"Estimated noise variance: {sigma2_hat}")
    print("Top correlation lag:", int(np.argmax(corrs)))
    print("Correlation first 15:", corrs[:15])
    
    # ---------------- PREPARE KALMANNET INPUT ----------------
    y_aligned = apply_delay_compensation(y_norm_full, d_hat)
    
    # prepare input
    y_input = y_aligned.astype(np.float32)[..., np.newaxis]
    y_input = y_input[np.newaxis, ...] # (1, T, 1)
    
    
    # reduce memory for KalmanNet training
    # max_knet_steps = min(2000, y_input.shape[1])
    # y_input_knet = y_input[:, :max_knet_steps, :]
    # y_aligned_knet = y_aligned[:max_knet_steps]
    y_aligned_tf = tf.convert_to_tensor(y_aligned, dtype=tf.float32)
    
    # ------------------------KALMANNET-----------------------
    knet = KalmanNetWrapper(config)
    optimizer = tf.keras.optimizers.Adam(5e-4)
    

    # ---------------- TRAIN KALMANNET ----------------
    print("\nTraining KalmanNet...\n")
    
    for epoch in range(10):

        with tf.GradientTape() as tape:
            x_hat_full, y_prior_full = knet(y_input, sigma2_hat, training=True)

            # remove batch dim
            y_prior_full = tf.squeeze(y_prior_full, axis=0)
            y_prior_full = tf.squeeze(y_prior_full, axis=-1)

            # SELF-SUPERVISED LOSS
            loss = tf.reduce_mean(tf.square(y_aligned_tf - y_prior_full), axis=None)

        grads = tape.gradient(loss, knet.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]
        optimizer.apply_gradients(zip(grads, knet.trainable_variables))

        print(f"Epoch {epoch+1} | KNet Loss: {loss.numpy():.4f}")
    
    # ---------------- FINAL INFERENCE ----------------
    x_hat_full, _ = knet(y_input, sigma2_hat, training=False)
    x_hat_full = x_hat_full.numpy().squeeze()
    
    
    # temporary comparison against observation target
    compare_len = min(len(x_hat_full), len(y_aligned))
    rmse = np.sqrt(
        np.mean((x_hat_full[-compare_len:] - y_aligned[-compare_len:]) ** 2)
    )
    
    print("\nKalmanNet temporary RMSE vs aligned observation:", rmse)
    




if __name__ == "__main__":
    main()