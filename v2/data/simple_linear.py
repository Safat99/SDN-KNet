import numpy as np


def generate_linear_signal(T=200, a=0.5, b=1.0):
    """
    Generate clean linear signal: x_t = a*t + b

    Args:
        T (int): number of timesteps
        a (float): slope
        b (float): intercept

    Returns:
        x_true (np.ndarray): shape (T,)
    """
    t = np.arange(T)
    x_true = a * t + b
    return x_true