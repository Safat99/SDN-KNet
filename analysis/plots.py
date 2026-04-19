import os
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, base_dir="reports/figures"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def plot_training(self, history, name="training", show=False):
        plt.figure()

        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Curve")
        plt.legend()

        save_path = os.path.join(self.base_dir, f"{name}_loss.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        plt.close()

    def plot_predictions(self, y_true, y_pred, name="pred", num_points=200, show=False):
        plt.figure(figsize=(10, 4))

        plt.plot(y_true[:num_points].squeeze(), label="True")
        plt.plot(y_pred[:num_points].squeeze(), label="Predicted")

        plt.title("Prediction vs Ground Truth")
        plt.legend()

        save_path = os.path.join(self.base_dir, f"{name}_prediction.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        plt.close()

    def plot_full_comparison(self, x_true, y_obs, y_pred, name="full", num_points=200, show=False):
        
        plt.figure(figsize=(10, 4))

        plt.plot(x_true[:num_points].squeeze(), label="Clean (x)")
        plt.plot(y_obs[:num_points].squeeze(), label="Corrupted (y)", alpha=0.7)
        plt.plot(y_pred[:num_points].squeeze(), label="Predicted (x̂)")

        plt.legend()
        plt.title("Clean vs Corrupted vs Prediction")

        save_path = os.path.join(self.base_dir, f"{name}_full.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()

        
        plt.close()