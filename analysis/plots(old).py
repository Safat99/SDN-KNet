import os
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, base_dir="reports/figures"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
        self.colors = {
            "blue": "#0072B2",
            "orange": "#E69F00",
            "green": "#009E73",
            "red": "#D55E00",
            "purple": "#CC79A7"
        }

    def _plot_line(self, data, label, color, linestyle, marker, zorder):
        plt.plot(
            data,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
            marker=marker,
            markevery=max(1, len(data)//20),  # sparse markers
            markersize=5,
            alpha=0.95,
            zorder=zorder
        )
    
    # ---------------- TRAINING CURVES ----------------
    def plot_training(self, history, name="training", show=False):

        # ---- LOSS (MSE) ----
        plt.figure()
        self._plot_line(
            history["train_loss"], label="Train MSE", 
            color=self.colors["blue"], 
            linestyle="-", 
            marker="o", 
            zorder=3
        )
        self._plot_line(
            history["val_loss"], label="Val MSE", 
            color=self.colors["orange"], 
            linestyle="--", 
            marker="s", 
            zorder=2
        )

        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training Curve (MSE)")
        plt.legend()

        save_path = os.path.join(self.base_dir, f"{name}_loss.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

        # ---- RMSE ----
        if "train_rmse" in history and "val_rmse" in history:
            plt.figure()
            self._plot_line(
                history["train_rmse"], label="Train RMSE",
                color=self.colors["green"],
                linestyle="-",
                marker="o",
                zorder=3
            )
            self._plot_line(
                history["val_rmse"], label="Val RMSE",
                color=self.colors["red"],
                linestyle="--",
                marker="s",
                zorder=2
            )

            plt.xlabel("Epoch")
            plt.ylabel("RMSE")
            plt.title("Training Curve (RMSE)")
            plt.legend()

            save_path = os.path.join(self.base_dir, f"{name}_rmse.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            plt.close()

        # ---- MAE ----
        if "train_mae" in history and "val_mae" in history:
            plt.figure()
            self._plot_line(
                history["train_mae"], label="Train MAE",
                color=self.colors["purple"],
                linestyle="-",
                marker="o",
                zorder=3
            )
            self._plot_line(
                history["val_mae"], label="Val MAE",
                color=self.colors["blue"],
                linestyle="--",
                marker="s",
                zorder=2
            )

            plt.xlabel("Epoch")
            plt.ylabel("MAE")
            plt.title("Training Curve (MAE)")
            plt.legend()

            save_path = os.path.join(self.base_dir, f"{name}_mae.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            plt.close()

    # ---------------- PREDICTIONS ----------------
    def plot_predictions(self, x_true, x_hat, name="pred", num_points=200, show=False):
        plt.figure(figsize=(10, 4))

        self._plot_line(
            x_true[:num_points].squeeze(), label="True",
            color=self.colors["blue"],
            linestyle="--",
            marker="o",
            zorder=3
        )
        self._plot_line(
            x_hat[:num_points].squeeze(), label="Estimated",
            color=self.colors["orange"],
            linestyle="-",
            marker="s",
            zorder=2
        )

        plt.xlabel("Time Step")
        plt.ylabel("Normalized Value")
        plt.title("State Estimation vs Ground Truth")
        plt.legend()

        save_path = os.path.join(self.base_dir, f"{name}_estimation.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        plt.close()

    # ---------------- FULL COMPARISON ----------------
    def plot_full_comparison(self, x_true, y_obs, x_hat, 
                             name="full", num_points=200, show=False, jitter=False):
        
        plt.figure(figsize=(10, 4))
        
        x = x_true[:num_points].squeeze()
        y = y_obs[:num_points].squeeze()
        xh = x_hat[:num_points].squeeze()
        
        # Optional tiny offset to visually separate overlapping lines
        if jitter:
            eps = 1e-4
            y = y + eps
            xh = xh - eps

        self._plot_line(
            x, label="Clean (x)",
            color=self.colors["blue"],
            linestyle="--",
            marker="o",
            zorder=4
        )
        self._plot_line(
            y, label="Corrupted (y)",
            color=self.colors["orange"],
            linestyle="-",
            marker="s",
            zorder=2
        )
        self._plot_line(
            xh, label="Estimated (x̂)",
            color=self.colors["green"],
            linestyle="-",
            marker="^",
            zorder=3
        )

        plt.xlabel("Time Step")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.title("Clean vs Corrupted vs Estimated")    

        save_path = os.path.join(self.base_dir, f"{name}_full.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()

        plt.close()