import numpy as np

# def estimate_delay(y, residual, max_delay=20):
#     """
#     Estimate delay using cross-correlation between residual and y
#     """
#     corrs = []
    
#     # smooth residual before correlation
#     residual = np.convolve(residual, np.ones(5)/5, mode='same')

#     for d in range(max_delay):
#         if d == 0:
#             r = residual
#             y_shift = y
#         else:
#             r = residual[d:]
#             y_shift = y[:-d]

#         corr = np.corrcoef(r, y_shift)[0,1]
#         corrs.append(corr)

#     d_hat = int(np.argmax(corrs))
#     return d_hat, corrs

def estimate_delay(y, signal, max_delay=20):
    corrs = []

    for d in range(max_delay):
        if d == 0:
            s = signal
            y_shift = y
        else:
            s = signal[d:]
            y_shift = y[:-d]

        corr = np.correlate(
            (s - np.mean(s)) / (np.std(s) + 1e-8),
            (y_shift - np.mean(y_shift)) / (np.std(y_shift) + 1e-8)
            )[0] / len(s)
        corrs.append(corr)

    d_hat = int(np.argmax(corrs))
    return d_hat, corrs


def apply_delay_compensation(y, d):
    if d == 0:
        return y

    y_shifted = np.zeros_like(y)
    y_shifted[d:] = y[:-d]

    return y_shifted