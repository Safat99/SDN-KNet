import numpy as np


def apply_delay(x, delay):
    """
    Apply fixed positive delay: y_t = x_{t-delay}

    Pads with initial value for simplicity.

    Args:
    x (np.ndarray): clean signal
    delay (int): delay steps

    Returns:
    x_delayed (np.ndarray)
    """
    if delay == 0:
        return x.copy()

    # pad = np.full(delay, x[0])
    # x_delayed = np.concatenate([pad, x[:-delay]])

    y = np.zeros_like(x)
    y[delay:] = x[:-delay]
    y[:delay] = x[0]  # pad

    return y


def add_gaussian_noise(x, std):
    """
    Add Gaussian noise

    Args:
    x (np.ndarray)
    std (float)

    Returns:
    noisy signal
    """
    rng = np.random.default_rng(seed=43)
    noise = rng.normal(0, std, size=x.shape)
    return x + noise


def corrupt_signal(x_true, delay=5, noise_std=0.5):
    """
    Full corruption pipeline

    Returns:
    y_obs (np.ndarray)
    metadata (dict)
    """
    x_delayed = apply_delay(x_true, delay)
    y_obs = add_gaussian_noise(x_delayed, noise_std)

    return y_obs, {
    "delay": delay,
    "noise_std": noise_std
    }