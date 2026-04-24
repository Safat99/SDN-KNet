import numpy as np


class LinearSSMGenerator:
    """
    Linear Gaussian State-Space Model

    x_t = A x_{t-1} + w_t
    y_t = H x_t + v_t
    """

    def __init__(
        self,
        A=0.9, # state transition matrix
        H=1.0, # observational matrix
        Q=0.01, # process noise co-variance matrix
        R=0.05, # measurement noise co-variance matrix 
        length=10000, # sequence length, doesn't have to be 10k sec
        x0=0.0,
    ):
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R
        self.length = length
        self.x0 = x0
        self.rng = np.random.default_rng(seed=42)

    def generate(self):
        x = np.zeros(self.length)
        y = np.zeros(self.length)

        x[0] = self.x0

        for t in range(1, self.length):
            w = self.rng.normal(0, np.sqrt(self.Q))
            v = self.rng.normal(0, np.sqrt(self.R))

            x[t] = self.A * x[t - 1] + w
            y[t] = self.H * x[t] + v

        return x, y