import numpy as np
from scipy.signal import correlation_lags


def estimate_delay(ref_signal, delayed_signal, max_delay=None):

    # Normalise for better peak detection (optional)
    ref_norm = (ref_signal - np.mean(ref_signal)) / (np.std(ref_signal) + 1e-8)
    delayed_norm = (delayed_signal - np.mean(delayed_signal)) / (np.std(delayed_signal) + 1e-8)

    # Full cross-correlation
    corr = np.correlate(delayed_norm, ref_norm, mode='full')
    lags = correlation_lags(len(delayed_norm), len(ref_norm), mode='full')    
    
    # Limit to a reasonable range if needed
    if max_delay is not None:
        mask = np.abs(lags) <= max_delay
        lags = lags[mask]
        corr = corr[mask]
    
    d_hat = lags[np.argmax(corr)]
    return d_hat, corr


# def apply_delay_compensation(y, d):
#     if d == 0:
#         return y

#     y_shifted = np.zeros_like(y)
#     y_shifted[d:] = y[:-d]

#     return y_shifted


def estimate_delay_mse(y_hat, y_obs, max_delay=20):
    best_delay = 0
    best_cost = float('inf')

    for d in range(-max_delay, max_delay + 1):
        if d >= 0:
            y_shift = y_hat[d:]
            y_ref = y_obs[:len(y_shift)]
        else:
            y_shift = y_hat[:d]
            y_ref = y_obs[-d:]

        cost = np.mean((y_shift - y_ref) ** 2)

        if cost < best_cost:
            best_cost = cost
            best_delay = d

    return best_delay

def compensate_delay(x, delay):
    if delay >= 0:
        return np.concatenate([x[delay:], np.zeros(delay)])
    else:
        return np.concatenate([np.zeros(-delay), x[:delay]])