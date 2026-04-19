import numpy as np


class MackeyGlassGenerator:
    """
    Mackey-Glass clean signal generator (discrete approximation)

    Parameters aligned with MGAB:
    tau=18, beta=0.25, gamma=0.1, n=10, history=0.9
    """

    def __init__(
        self,
        tau=18,
        beta=0.25,
        gamma=0.1,
        n=10,
        history=0.9,
        dt=1.0,
        length=10000,
        burn_in=1000,
    ):
        self.tau = tau
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.history = history
        self.dt = dt
        self.length = length
        self.burn_in = burn_in

    def generate(self):
        delay_steps = int(self.tau / self.dt)

        total_length = self.length + self.burn_in + delay_steps
        x = np.zeros(total_length)

        # initial history
        x[:delay_steps] = self.history

        for t in range(delay_steps, total_length):
            x_tau = x[t - delay_steps]
            x_now = x[t - 1]

            dx = (
                self.beta * x_tau / (1 + x_tau ** self.n)
                - self.gamma * x_now
            )

            x[t] = x_now + dx * self.dt

        # remove transient (VERY IMPORTANT)
        return x[delay_steps + self.burn_in : delay_steps + self.burn_in + self.length]