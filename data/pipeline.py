import numpy as np

from data.generators.mackey_glass import MackeyGlassGenerator
from data.generators.linear_ssm import LinearSSMGenerator
from data.corruption.delay import DelayCorruptor
from data.corruption.noise import NoiseCorruptor


class TimeSeriesPipeline:
    def __init__(self, config):
        self.config = config

        # --- Generator ---
        dataset_name = config["dataset"]["name"]

        if dataset_name == "mackey_glass":
            self.generator = MackeyGlassGenerator(
                tau=config["dataset"]["tau"],
                length=config["dataset"]["sequence_length"]
            )

        elif dataset_name == "linear":
            self.generator = LinearSSMGenerator(
                A=config["dataset"]["A"],
                H=config["dataset"]["H"],
                Q=config["dataset"]["Q"],
                R=config["dataset"]["R"],
                length=config["dataset"]["sequence_length"]
            )

        # --- Corruptions ---
        self.delay = DelayCorruptor(config["corruption"]["delay"])
        self.noise = NoiseCorruptor(config["corruption"]["noise_std"])

        # --- Window params ---
        self.window_size = config["window"]["input_length"]
        # self.horizon = config["window"]["forecast_horizon"]

    # -----------------------------
    # Step 1: Generate clean data
    # -----------------------------
    # def generate_clean(self):
    #     if self.config["dataset"]["name"] == "linear":
    #         x, _ = self.generator.generate()
    #         return x
    #     else:
    #         return self.generator.generate()
    
    def generate_clean(self):
        dataset_name = self.config["dataset"]["name"]

        # =========================
        # LINEAR (unchanged)
        # =========================
        if dataset_name == "linear":
            x, _ = self.generator.generate()
            return x

        # =========================
        # MG or other synthetic (unchanged)
        # =========================
        elif dataset_name in ["mg", "mackey_glass"]:
            return self.generator.generate()

        # =========================
        # JENA (NEW)
        # =========================
        elif dataset_name == "jena":
            import pandas as pd

            path = self.config["dataset"]["path"]
            column = self.config["dataset"]["target_column"]

            df = pd.read_csv(path)

            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in dataset")

            y = df[column].values.astype(np.float32)

            print("Loaded Jena data:", y.shape)

            return y

        # =========================
        # FALLBACK (safety)
        # =========================
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    # -----------------------------
    # Step 2: Apply corruption
    # -----------------------------
    def corrupt(self, x):
        y = self.delay.apply(x)
        y = self.noise.apply(y)
        return y

    # -----------------------------
    # Step 3: Split RAW sequences
    # -----------------------------
    def split(self, x, y):
        n = len(x)

        train_end = int(n * self.config["split"]["train"])
        val_end = int(n * (self.config["split"]["train"] + self.config["split"]["val"]))

        return {
            "train": (x[:train_end], y[:train_end]),
            "val": (x[train_end:val_end], y[train_end:val_end]),
            "test": (x[val_end:], y[val_end:])
        }
        
        
    # -----------------------------
    # Step 4: Normalize using TRAIN stats only
    # -----------------------------
    def normalize(self, splits):
        _, y_train = splits["train"] # x_train will not be needed in normalizing

        mean = np.mean(y_train)
        std = np.std(y_train) + 1e-8

        norm_splits = {}

        for key in splits:
            x, y = splits[key]
            x_norm = (x - mean) / std
            y_norm = (y - mean) / std
            norm_splits[key] = (x_norm, y_norm)

        return norm_splits, mean, std

    # -----------------------------
    # Step 5: Create windows 
    # Input will be the corrupted observations(y) and Target will be the clean signal (x) )
    # -----------------------------
    def create_windows(self, x, y):
        X, Y = [], []

        for i in range(len(x) - self.window_size):
            X.append(y[i : i + self.window_size])
            Y.append(x[i + self.window_size - 1]) 

        return np.array(X), np.array(Y)


    # -----------------------------
    # FULL PIPELINE
    # -----------------------------
    def run(self):
        # clean signal (ground truth)
        x = self.generate_clean()

        # corrupted observation
        y = self.corrupt(x)
        
        # split first
        splits = self.split(x, y)

        # normalize using train stats
        norm_splits, mean, std = self.normalize(splits)

        # window per split
        data = {}
        for key in norm_splits:
            x_norm, y_norm = norm_splits[key]
            X, Y = self.create_windows(x_norm, y_norm)
            data[key] = (X, Y)

        # full normalized sequences (for plotting)
        x_norm_full = (x - mean) / std
        y_norm_full = (y - mean) / std

        return data, (mean, std), x_norm_full, y_norm_full

    
    def create_ssl_windows(self, y):
        """
        Self-supervised windowing:
        input  = y[t-W : t]
        target = y[t]
        """
        X, Y = [], []

        for i in range(len(y) - self.window_size):
            X.append(y[i : i + self.window_size])
            Y.append(y[i + self.window_size])

        return np.array(X), np.array(Y)
    
    # run self supervised pipeline     
    def run_ssl(self):
        x = self.generate_clean()
        y = self.corrupt(x)

        # split raw
        splits = self.split(x, y)

        # normalize
        norm_splits, mean, std = self.normalize(splits)

        # SSL windows (y → y)
        data = {}
        for key in norm_splits:
            _, y_norm = norm_splits[key]
            X, Y = self.create_ssl_windows(y_norm)
            data[key] = (X, Y)
        
        # full normalized for plotting/debug
        y_norm_full = (y - mean) / std

        return data, (mean, std), y_norm_full