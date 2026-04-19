import numpy as np


class DelayCorruptor:
    def __init__(self, delay=0):
        self.delay = delay

    def apply(self, x):
        if self.delay == 0:
            return x.copy()

        y = np.zeros_like(x)
        y[self.delay:] = x[:-self.delay]
        y[:self.delay] = x[0]  # pad

        return y