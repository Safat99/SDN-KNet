import numpy as np


class NoiseCorruptor:
    def __init__(self, noise_std=0.0):
        self.noise_std = noise_std

    def apply(self, x):
        noise = np.random.normal(0, self.noise_std, size=len(x))
        return x + noise