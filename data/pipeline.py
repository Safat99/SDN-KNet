import numpy as np

from data.generators.mackey_glass import MackeyGlassGenerator
from data.corruption.delay import DelayCorruptor
from data.corruption.noise import NoiseCorruptor


class TimeSeriesPipeline:
    def __init__(self, config):
        self.config = config

        # --- Generator ---
        self.generator = MackeyGlassGenerator(
            tau=config["dataset"]["tau"],
            length=config["dataset"]["sequence_length"]
        )

        # --- Corruptions ---
        self.delay = DelayCorruptor(config["corruption"]["delay"])
        self.noise = NoiseCorruptor(config["corruption"]["noise_std"])

        # --- Window params ---
        self.window_size = config["window"]["input_length"]
        self.horizon = config["window"]["forecast_horizon"]

    # -----------------------------
    # Step 1: Generate clean data
    # -----------------------------
    def generate_clean(self):
        x = self.generator.generate()
        return x

    # -----------------------------
    # Step 2: Apply corruption
    # -----------------------------
    def corrupt(self, x):
        y = self.delay.apply(x)
        y = self.noise.apply(y)
        return y

    # -----------------------------
    # Step 3: Normalize (important)
    # -----------------------------
    def normalize(self, x):
        mean = np.mean(x)
        std = np.std(x) + 1e-8
        return (x - mean) / std, mean, std

    # -----------------------------
    # Step 4: Create windows
    # -----------------------------
    def create_windows(self, x, y):
        X, Y = [], []

        for i in range(len(x) - self.window_size - self.horizon):
            X.append(y[i : i + self.window_size])
            Y.append(x[i + self.window_size : i + self.window_size + self.horizon])

        return np.array(X), np.array(Y)

    # -----------------------------
    # Step 5: Train/Val/Test split
    # -----------------------------
    def split(self, X, Y):
        n = len(X)
        train_end = int(n * self.config["split"]["train"])
        val_end = int(n * (self.config["split"]["train"] + self.config["split"]["val"]))

        return {
            "train": (X[:train_end], Y[:train_end]),
            "val": (X[train_end:val_end], Y[train_end:val_end]),
            "test": (X[val_end:], Y[val_end:])
        }

    # -----------------------------
    # FULL PIPELINE
    # -----------------------------
    def run(self):
        # clean signal (ground truth)
        x = self.generate_clean()

        # corrupted observation
        y = self.corrupt(x)

        # normalize using observation (realistic)
        y_norm, mean, std = self.normalize(y)
        x_norm = (x - mean) / std

        # windowing
        X, Y = self.create_windows(x_norm, y_norm)

        # split
        data = self.split(X, Y)

        return data, (mean, std), x_norm, y_norm