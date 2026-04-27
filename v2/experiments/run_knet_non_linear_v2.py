import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from v2.data.simple_linear import generate_linear_signal
from v2.data.corrupt import corrupt_signal
from data.generators.linear_ssm import LinearSSMGenerator
from data.generators.mackey_glass import MackeyGlassGenerator

from models.gru_baseline import GRUBaseline
from models.gru_encoder_cho import ChoGRUEncoderPredictor
from utils.delay_estimation import estimate_delay, estimate_delay_mse
from utils.noise_estimation import estimate_noise_variance

from models.kalmannet_wrapper import KalmanNetWrapper

def rmse(x_hat, x_true):
    return np.sqrt(np.mean((x_hat - x_true) ** 2))


def moving_average_denoise(y, window=10):
    """
    Simple baseline denoiser (no delay handling yet)
    """
    y_padded = np.pad(y, (window//2, window//2), mode='edge')
    y_smooth = np.convolve(y_padded, np.ones(window)/window, mode='valid')
    return y_smooth

def compensate_delay(x, delay):
    if delay >= 0:
        return np.concatenate([x[delay:], np.zeros(delay)]) # left shift
    else:
        return np.concatenate([np.zeros(-delay), x[:delay]])


def create_self_supervised_dataset(y_obs, window_size=20):
    """Creates sliding windows for self-supervised learning."""
    X, Y = [], []

    for i in range(len(y_obs) - window_size):
        X.append(y_obs[i:i+window_size])
        Y.append(y_obs[i+window_size])  # next value

    X = np.array(X)[..., np.newaxis]
    Y = np.array(Y)

    return X, Y

def split_sequential(X, Y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Splits data sequentially (no shuffling) to respect time order."""
    total = len(X)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
    X_test, Y_test = X[val_end:], Y[val_end:]
    
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)

def train_gru(X_train, Y_train, X_val, Y_val, hidden_size=64, output_dim=1, epochs=5):
    """Trains GRU model with optional validation."""
    
    # model = GRUBaseline(hidden_size=64)
    
    model = ChoGRUEncoderPredictor(hidden_size=output_dim, output_dim=output_dim)
    Y_train = Y_train[..., None]   # make shape (batch, 1) when running the Cho's model
    Y_val = Y_val[..., None] # for baseline running these two lines don't need
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001), # TODO 0.0001 will have to use finally
        loss='mse'
    )

    # Use validation data for monitoring (optional early stopping)
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=64,
        verbose=1
    )
    return model, history


def predict_on_windows(model, y_obs, window_size=30):
    """Predicts next values for each window in the sequence."""
    X = []
    
    for i in range(len(y_obs) - window_size):
        X.append(y_obs[i:i+window_size])
    
    X = np.array(X)[..., np.newaxis]
    
    preds = model.predict(X, verbose=0).flatten()
    
    # Align predictions with original time axis (first 'window_size' values are placeholder)
    y_hat_norm = np.concatenate([
        np.full(window_size, preds[0]),
        preds
    ])
    
    return y_hat_norm


def standardize(x):
    mu = np.mean(x)
    sigma = np.std(x) + 1e-8
    return (x - mu) / sigma, mu, sigma

def unstandardize(x, mu, sigma):
    return x * sigma + mu


def apply_delay(y_seq, delay):
    if delay > 0:
        return y_seq[:, :-delay, :]
    elif delay < 0:
        return y_seq[:, -delay:, :]
    return y_seq


def run_kalmannet_inference(knet ,y_obs_test, delay, noise_var):

    
    y_norm, _, _ = standardize(y_obs_test)
    
    # convert to tensor
    y_seq = tf.convert_to_tensor(
        y_norm[None, :, None],
        dtype=tf.float32
    )

    # apply delay
    y_seq = apply_delay(y_seq, delay)

    # ensure float32
    noise_var = np.float32(noise_var)

    # forward pass
    x_hat_seq, _ = knet(y_seq, sigma2_hat=noise_var)

    x_hat = x_hat_seq.numpy().flatten()

    return x_hat


def train_kalmannet(y_obs_train, delay, noise_var, epochs=10):

    config = {
        "dataset": {
            "A": 1.0,
            "H": 1.0
        },
        "model": {
            "hidden_size": 64
        }
    }

    y_norm, mu, sigma = standardize(y_obs_train)

    y_seq = tf.convert_to_tensor(
        y_norm[None, :, None],
        dtype=tf.float32
    )
    

    y_seq = apply_delay(y_seq, delay)
    
    noise_var = np.float32(noise_var)
    
    knet = KalmanNetWrapper(config, return_prior=True)
    
    optimizer = tf.keras.optimizers.Adam(1e-4)


    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            _, y_prior_seq = knet(y_seq, sigma2_hat=noise_var)
            
            innovation = y_seq - y_prior_seq
            
            tf.print("innovation mean:", tf.reduce_mean(tf.abs(innovation)))
            
            # -------- loss --------
            mse_loss = tf.reduce_mean(tf.square(innovation))

            l2_reg = tf.add_n([
                tf.nn.l2_loss(w) for w in knet.trainable_variables
                if "kernel" in w.name
            ])
            
            gain_penalty = tf.reduce_mean(tf.square(y_prior_seq))
            
            loss = mse_loss + 1e-4 * l2_reg + 1e-3 * gain_penalty

        grads = tape.gradient(loss, knet.trainable_variables)
        optimizer.apply_gradients(zip(grads, knet.trainable_variables))

        print(f"[KalmanNet] Epoch {epoch+1}, Loss: {loss.numpy():.4f}")

    return knet


################################################## main #############################################
def main():
    np.random.seed(0)

    # --------------------------
    # 1. Generate clean signal
    # --------------------------
    generator = MackeyGlassGenerator(dt=0.1, length=10000)
    x_true  = generator.generate()

    

    # --------------------------
    # 2. Corrupt it
    # --------------------------
    y_obs, meta = corrupt_signal(
        x_true,
        delay=5,
        noise_std=10.0
    )

    print("True delay:", meta["delay"])
    print("Noise std:", meta["noise_std"])
    
    # --------------------------
    # 3. Normalize
    # --------------------------
    y_norm, y_mu, y_sigma = standardize(y_obs)

    # --------------------------
    # 4. Create windowed dataset
    # --------------------------
    window_size = 300
    X, Y = create_self_supervised_dataset(y_norm, window_size)

    # --------------------------
    # 5. Sequential split (train/val/test)
    # --------------------------
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_sequential(
        X, Y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

    # --------------------------
    # 6. Train model
    # --------------------------
    model, history = train_gru(X_train, Y_train, X_val, Y_val, epochs=10)

    # --------------------------
    # 7. Predict on test set
    # --------------------------
    start_test_idx = len(X_train) + len(X_val) # first index of test window in original signal
    y_norm_test = y_norm[start_test_idx:]
    y_hat_norm_test = predict_on_windows(model, y_norm_test, window_size)

    # convert back to original scale
    y_hat_test = unstandardize(y_hat_norm_test, y_mu, y_sigma)
    
    # The ground truth clean signal for the test segment (same time indices)
    x_true_test = x_true[start_test_idx: start_test_idx + len(y_hat_test)]
    # Observed (corrupted) signal for test segment
    y_obs_test = y_obs[start_test_idx: start_test_idx + len(y_hat_test)]
    
    
    # --------------------------
    # 8. Delay estimation (only on test data)
    # --------------------------
    # Use residual of test segment
    residual_test = y_obs_test - y_hat_test
    
    estimated_delay, _ = estimate_delay(ref_signal=residual_test, delayed_signal=y_obs_test, max_delay=20)
    # estimated_delay = estimate_delay_mse(y_hat, y_obs, max_delay=20)
    
    y_hat_test_compensated = compensate_delay(y_hat_test, estimated_delay)
    print(f"estimated delay found: {estimated_delay}")

    # --------------------------
    # 9. Noise variance estimation on test residual
    # --------------------------
    noise_var_test = estimate_noise_variance(residual_test)
    print(f"estimated_noise_found: {noise_var_test:.4f}")
    print(f"estimated_noise_std: {np.sqrt(noise_var_test):.4f}")

    # --------------------------
    # 10. KalmanNet inference
    # --------------------------
    y_obs_train = y_obs[:start_test_idx]
    # y_obs_train = y_obs[:300]
    
    knet = train_kalmannet(
        y_obs_train,
        estimated_delay,
        noise_var_test,
        epochs=50
    )
    
    x_hat_test = run_kalmannet_inference(
        knet,
        y_obs_test,
        estimated_delay,
        noise_var_test
    )

        
    # --------------------------
    # 11. Evaluation (synthetic test set only!)
    # --------------------------
    rmse_obs_test = rmse(y_obs_test, x_true_test)
    rmse_hat_test = rmse(y_hat_test_compensated, x_true_test)
    rmse_knet = rmse(x_hat_test, x_true_test[:len(x_hat_test)])

    print(f"RMSE (observed vs true): {rmse_obs_test:.4f}")
    print(f"RMSE (denoised vs true): {rmse_hat_test:.4f}")
    print(f"RMSE (KalmanNet vs true): {rmse_knet:.4f}")
    
    # --------------------------
    # 12. Plot
    # --------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(x_true_test[:200], label="x_true (clean)", linewidth=2)
    plt.plot(y_obs_test[:200], label="y_obs (corrupted)", alpha=0.8)
    plt.plot(y_hat_test_compensated[:200], label="y_hat (estimated)",  linewidth=2, color='black')
    plt.plot(x_hat_test[:200], label="x_hat (KalmanNet)", linewidth=2, color='red')

    plt.legend()
    plt.title("Mackey Glass Signal Denoising")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

    # plt.show()
    plt.savefig('non_linear_v2_knet_more_training_samples.png')


if __name__ == "__main__":
    main()