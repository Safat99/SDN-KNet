import numpy as np

def estimate_noise_variance(residual): 
    dr = residual[1:] - residual[:-1] # Δr(t)=r(t)−r(t−1),
    sigma2 = np.var(dr) / 2.0
    return sigma2