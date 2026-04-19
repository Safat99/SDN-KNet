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

    def save_predictions(self, y_true, y_pred, name):
        path = os.path.join(self.base_dir, f"{name}_predictions.npz")

        np.savez(path, y_true=y_true, y_pred=y_pred)

    def save_metrics(self, metrics_dict, name):
        path = os.path.join(self.base_dir, f"{name}_metrics.json")

        with open(path, "w") as f:
            json.dump(metrics_dict, f)