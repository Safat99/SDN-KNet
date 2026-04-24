import numpy as np

def estimate_delay(y, residual, max_delay=20):
    """
    Estimate delay using cross-correlation between residual and y
    """
    corrs = []
    
    # smooth residual before correlation
    residual = np.convolve(residual, np.ones(5)/5, mode='same')

    for d in range(max_delay):
        if d == 0:
            # corr = np.mean(residual * y)
            corr = np.corrcoef(residual * y)[0,1]
        else:
            # corr = np.mean(residual[d:] * y[:-d])
            corr = np.corrcoef(residual[d:], y[:-d])[0,1]

        corrs.append(corr)

    d_hat = int(np.argmax(corrs))
    return d_hat, corrs


def apply_delay_compensation(y, d):
    if d == 0:
        return y

    y_shifted = np.zeros_like(y)
    y_shifted[d:] = y[:-d]

    return y_shifted