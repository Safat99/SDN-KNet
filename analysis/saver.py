import os
import json
import numpy as np


class ResultSaver:
    def __init__(self, base_dir="results"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save_history(self, history, name):
        path = os.path.join(self.base_dir, f"{name}_history.json")

        with open(path, "w") as f:
            json.dump(history, f)

    def save_predictions(self, x_true, x_hat, y_input, name):
        path = os.path.join(self.base_dir, f"{name}_estimations.npz")

        np.savez(path, x_true=x_true, x_hat=x_hat, y_input = y_input)

    def save_metrics(self, metrics_dict, name):
        path = os.path.join(self.base_dir, f"{name}_metrics.json")

        with open(path, "w") as f:
            json.dump(metrics_dict, f)