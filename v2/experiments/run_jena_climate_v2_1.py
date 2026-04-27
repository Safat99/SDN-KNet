import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from models.gru_baseline import GRUBaseline
from utils.noise_estimation import estimate_noise_variance
from models.kalmannet_wrapper import KalmanNetWrapper


# --------------------------
# Data
# --------------------------
def load_jena_temperature(path, max_points=5000):
    df = pd.read_csv(path)
    return df["T (degC)"].values.astype(float)[:max_points]


# --------------------------
# Masking
# --------------------------
def mask_signal(y, mask_ratio=0.1):
    y_masked = y.copy()
    n = len(y)
    mask_indices = np.random.choice(n, int(n * mask_ratio), replace=False)
    y_masked[mask_indices] = np.nan
    return y_masked, mask_indices


def fill_missing(y):
    y_filled = y.copy()
    isnan = np.isnan(y_filled)
    y_filled[isnan] = np.interp(
        np.flatnonzero(isnan),
        np.flatnonzero(~isnan),
        y_filled[~isnan]
    )
    return y_filled


# --------------------------
# Utils
# --------------------------
def standardize(x):
    mu = np.mean(x)
    sigma = np.std(x) + 1e-8
    return (x - mu) / sigma, mu, sigma


def unstandardize(x, mu, sigma):
    return x * sigma + mu


# --------------------------
# Dataset
# --------------------------
def create_dataset(y, window=50):
    X, Y = [], []
    for i in range(len(y) - window):
        X.append(y[i:i+window])
        Y.append(y[i+window])
    return np.array(X)[..., None], np.array(Y)


def split_data(X, Y):
    n = len(X)
    t1 = int(0.7 * n)
    t2 = int(0.85 * n)
    return (X[:t1], Y[:t1]), (X[t1:t2], Y[t1:t2]), (X[t2:], Y[t2:])


# --------------------------
# GRU
# --------------------------
def train_gru(X_train, Y_train, X_val, Y_val):
    model = GRUBaseline(hidden_size=64)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="mse"
    )

    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=10,
        batch_size=64,
        verbose=1
    )

    return model


def predict_sequence(model, y, window):
    X = [y[i:i+window] for i in range(len(y) - window)]
    X = np.array(X)[..., None]

    preds = model.predict(X, verbose=0).flatten()

    return np.concatenate([np.full(window, preds[0]), preds])


# --------------------------
# KalmanNet (STABLE)
# --------------------------
def train_kalmannet(y_train, noise_var):

    y_norm, _, _ = standardize(y_train)

    y_seq = tf.convert_to_tensor(
        y_norm[None, :, None],
        dtype=tf.float32
    )

    config = {
        "dataset": {"A": 0.98, "H": 1.0},
        "model": {"hidden_size": 64}
    }

    knet = KalmanNetWrapper(config, return_prior=True)
    opt = tf.keras.optimizers.Adam(1e-4)

    for epoch in range(30):
        with tf.GradientTape() as tape:
            _, y_prior = knet(y_seq, sigma2_hat=np.float32(noise_var))
            innovation = y_seq - y_prior
            loss = tf.reduce_mean(tf.square(innovation))

        grads = tape.gradient(loss, knet.trainable_variables)

        grad_var_pairs = [
            (tf.clip_by_norm(g, 1.0), v)
            for g, v in zip(grads, knet.trainable_variables)
            if g is not None
        ]

        opt.apply_gradients(grad_var_pairs)

        print(f"[KalmanNet] Epoch {epoch+1}, Loss: {loss.numpy():.4f}")

    return knet


def run_kalmannet(knet, y_test, noise_var):
    y_norm, _, _ = standardize(y_test)

    y_seq = tf.convert_to_tensor(
        y_norm[None, :, None],
        dtype=tf.float32
    )

    x_hat, _ = knet(y_seq, sigma2_hat=np.float32(noise_var))

    return x_hat.numpy().flatten()


# --------------------------
# MAIN
# --------------------------
def main():

    np.random.seed(0)

    # 1. Load data
    y_obs = load_jena_temperature("data/jena_climate_2009_2016.csv")

    # 2. Mask
    y_masked, mask_idx = mask_signal(y_obs, 0.1)

    # 3. Fill
    y_filled = fill_missing(y_masked)

    # 4. Normalize
    y_norm, mu, sigma = standardize(y_filled)

    # 5. Dataset
    window = 50
    X, Y = create_dataset(y_norm, window)

    (X_tr, Y_tr), (X_val, Y_val), (X_te, Y_te) = split_data(X, Y)

    # 6. Train GRU
    model = train_gru(X_tr, Y_tr, X_val, Y_val)

    # 7. Estimate signal
    start = len(X_tr) + len(X_val) + window

    y_test_norm = y_norm[start:]
    y_hat_norm = predict_sequence(model, y_test_norm, window)
    y_hat = unstandardize(y_hat_norm, mu, sigma)

    y_test = y_obs[start:start + len(y_hat)]

    # 8. Masked RMSE
    mask_test = mask_idx[
        (mask_idx >= start) &
        (mask_idx < start + len(y_hat))
    ]

    mask_local = mask_test - start

    if len(mask_local) > 0:
        rmse_masked = np.sqrt(
            np.mean((y_hat[mask_local] - y_obs[mask_test])**2)
        )
        print(f"Masked RMSE: {rmse_masked:.4f}")

    # 9. Noise estimate (STABILIZED)
    residual = y_test - y_hat
    noise_var = estimate_noise_variance(residual)
    noise_var = max(noise_var, 0.1)

    print(f"Noise var used: {noise_var:.4f}")

    # 10. KalmanNet (short sequence only!)
    knet = train_kalmannet(y_obs[:1000], noise_var)
    x_hat = run_kalmannet(knet, y_test, noise_var)

    # 11. Plot
    plt.figure(figsize=(12,5))
    plt.plot(y_test[:200], label="Observed", alpha=0.5)
    plt.plot(y_hat[:200], label="GRU Estimate", linewidth=2)
    plt.plot(x_hat[:200], label="KalmanNet Estimate", linewidth=2)

    plt.legend()
    plt.title("Jena Climate Estimation")
    plt.grid(True)
    plt.savefig('jena_climate_estimation.png')


if __name__ == "__main__":
    main()