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

    # ---------------- GRU ENCODER ----------------
    X_train = X_train.astype(np.float32)[..., np.newaxis]
    Y_train = Y_train.astype(np.float32)

    X_val = X_val.astype(np.float32)[..., np.newaxis]
    Y_val = Y_val.astype(np.float32)

    X_test = X_test.astype(np.float32)[..., np.newaxis]
    Y_test = Y_test.astype(np.float32)
 
    model = GRUBaseline(
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        output_dim=1
    )

    # ----------------- TRAIN -----------------------
    trainer = Trainer(model, config)
    trainer.train((X_train, Y_train), (X_val, Y_val))

    # ---------------- PREDICT y_hat ----------------
    y_hat = model(X_test, training=False).numpy().squeeze()
    y_true = Y_test.squeeze()

    # ---------------- RESIDUAL ----------------
    residual = y_true - y_hat

    print("Residual mean:", np.mean(residual))
    print("Residual variance:", np.var(residual))
    print("Residual shape:", residual.shape)
    
    # ---------------- ESTIMATION ----------------
    d_hat, _ = estimate_delay(y_true, residual, max_delay=20)
    sigma2_hat = estimate_noise_variance(residual)
    
    print(f"Estimated delay: {d_hat}")
    print(f"Estimated noise variance: {sigma2_hat}")
    
    # ---------------- PREPARE KALMANNET INPUT ----------------
    y_aligned = apply_delay_compensation(y_norm_full, d_hat)
    
    # prepare input
    y_input = y_aligned.astype(np.float32)[..., np.newaxis]
    y_input = y_input[np.newaxis, ...]
    
    
    # ------------------------KALMANNET-----------------------
    knet = KalmanNetWrapper(config)
    
    optimizer = tf.keras.optimizers.Adam(1e-3)
    
    y_aligned_tf = tf.convert_to_tensor(y_aligned, dtype=tf.float32)

    # ---------------- TRAIN KALMANNET ----------------
    print("\nTraining KalmanNet...\n")
    
    for epoch in range(10):

        with tf.GradientTape() as tape:
            x_hat_full, y_prior_full = knet(y_input, sigma2_hat, training=True)

            # remove batch dim
            y_prior_full = tf.squeeze(y_prior_full, axis=0)

            # SELF-SUPERVISED LOSS
            loss = tf.reduce_mean(tf.square(y_aligned_tf - y_prior_full), axis=None)

        grads = tape.gradient(loss, knet.trainable_variables)
        optimizer.apply_gradients(zip(grads, knet.trainable_variables))

        print(f"Epoch {epoch+1} | KNet Loss: {loss.numpy():.4f}")
    
    # ---------------- FINAL INFERENCE ----------------
    x_hat_full = knet(y_input, sigma2_hat, training=True)
    x_hat_full = x_hat_full.numpy().squeeze()
    
    # align with test
    x_hat = x_hat_full[-len(y_true):]
    
    # TEMP evaluation (state vs observation)
    rmse = np.sqrt(np.mean((x_hat - y_true)**2))
    
    print("\nKalmanNet RMSE:", rmse)
    




if __name__ == "__main__":
    main()