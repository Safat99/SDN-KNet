import os

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, base_dir="reports/figures"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        self.colors = {
            "blue": "#0072B2",
            "orange": "#E69F00",
            "green": "#009E73",
            "red": "#D55E00",
            "purple": "#CC79A7",
            "gray": "#6C757D",
        }

    def _to_1d(self, data, num_points=None):
        array = np.asarray(data).squeeze()
        if num_points is not None:
            array = array[:num_points]
        return np.ravel(array)

    def _new_figure(self, figsize=(10, 4.8), nrows=1, height_ratios=None, sharex=False):
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            figsize=figsize,
            sharex=sharex,
            gridspec_kw={"height_ratios": height_ratios} if height_ratios else None,
            constrained_layout=True,
        )
        return fig, axes

    def _style_axes(self, ax, xlabel=None, ylabel=None, title=None):
        ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.margins(x=0.01)

    def _save_figure(self, fig, save_path, show=False):
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def _plot_line(
        self,
        ax,
        x,
        y,
        label,
        color,
        linestyle="-",
        linewidth=2.2,
        alpha=0.95,
        zorder=2,
        marker=None,
    ):
        x = np.asarray(x)
        y = np.asarray(y)

        use_marker = marker is not None and len(y) <= 40
        markevery = max(1, len(y) // 15) if use_marker else None
        markersize = 4 if use_marker else 0

        ax.plot(
            x,
            y,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
            marker=marker if use_marker else None,
            markevery=markevery,
            markersize=markersize,
        )

    def _set_value_padding(self, ax, series_list, pad_ratio=0.08):
        values = [np.asarray(series).ravel() for series in series_list if len(np.asarray(series).ravel()) > 0]
        if not values:
            return

        stacked = np.concatenate(values)
        ymin = float(np.min(stacked))
        ymax = float(np.max(stacked))

        if np.isclose(ymin, ymax):
            pad = max(abs(ymin) * 0.1, 1e-3)
        else:
            pad = (ymax - ymin) * pad_ratio

        ax.set_ylim(ymin - pad, ymax + pad)

    def _should_use_log_scale(self, values):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        arr = arr[arr > 0]

        if len(arr) < 2:
            return False

        low = np.percentile(arr, 10)
        high = np.percentile(arr, 100)
        if low <= 0:
            return False

        return (high / low) >= 20

    def _plot_metric_pair(self, history, train_key, val_key, ylabel, title, filename, colors, show):
        if train_key not in history or val_key not in history:
            return

        train_values = self._to_1d(history[train_key])
        val_values = self._to_1d(history[val_key])
        epochs = np.arange(1, len(train_values) + 1)

        fig, ax = self._new_figure(figsize=(9, 4.8))

        self._plot_line(
            ax,
            epochs,
            train_values,
            label=f"Train {ylabel}",
            color=colors[0],
            linestyle="-",
            zorder=3,
            marker="o",
        )
        self._plot_line(
            ax,
            epochs,
            val_values,
            label=f"Val {ylabel}",
            color=colors[1],
            linestyle="--",
            zorder=2,
            marker="s",
        )

        if self._should_use_log_scale(np.concatenate([train_values, val_values])):
            ax.set_yscale("log")

        best_epoch = int(np.argmin(val_values))
        ax.scatter(
            epochs[best_epoch],
            val_values[best_epoch],
            color=colors[1],
            s=40,
            zorder=5,
        )
        ax.annotate(
            f"best val @ {epochs[best_epoch]}",
            xy=(epochs[best_epoch], val_values[best_epoch]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color=colors[1],
        )

        self._style_axes(ax, xlabel="Epoch", ylabel=ylabel, title=title)
        ax.legend(frameon=True)

        save_path = os.path.join(self.base_dir, filename)
        self._save_figure(fig, save_path, show=show)

    # ---------------- TRAINING CURVES ----------------
    def plot_training(self, history, name="training", show=False):
        self._plot_metric_pair(
            history,
            train_key="train_loss",
            val_key="val_loss",
            ylabel="MSE",
            title="Training Curve (MSE)",
            filename=f"{name}_loss.png",
            colors=(self.colors["blue"], self.colors["orange"]),
            show=show,
        )

        self._plot_metric_pair(
            history,
            train_key="train_rmse",
            val_key="val_rmse",
            ylabel="RMSE",
            title="Training Curve (RMSE)",
            filename=f"{name}_rmse.png",
            colors=(self.colors["green"], self.colors["red"]),
            show=show,
        )

        self._plot_metric_pair(
            history,
            train_key="train_mae",
            val_key="val_mae",
            ylabel="MAE",
            title="Training Curve (MAE)",
            filename=f"{name}_mae.png",
            colors=(self.colors["purple"], self.colors["blue"]),
            show=show,
        )

    # ---------------- PREDICTIONS ----------------
    def plot_predictions(self, x_true, x_hat, name="pred", num_points=200, show=False):
        true_values = self._to_1d(x_true, num_points=num_points)
        pred_values = self._to_1d(x_hat, num_points=num_points)
        time_index = np.arange(len(true_values))
        residual = pred_values - true_values

        fig, axes = self._new_figure(
            figsize=(11, 6),
            nrows=2,
            height_ratios=[3, 1],
            sharex=True,
        )
        ax_main, ax_res = axes

        self._plot_line(
            ax_main,
            time_index,
            true_values,
            label="True",
            color=self.colors["blue"],
            linestyle="--",
            linewidth=2.4,
            alpha=0.95,
            zorder=3,
        )
        self._plot_line(
            ax_main,
            time_index,
            pred_values,
            label="Estimated",
            color=self.colors["orange"],
            linestyle="-",
            linewidth=2.0,
            alpha=0.9,
            zorder=2,
        )

        error_mae = float(np.mean(np.abs(residual)))
        ax_main.fill_between(
            time_index,
            true_values,
            pred_values,
            color=self.colors["orange"],
            alpha=0.12,
            zorder=1,
            label=f"|error| area (MAE={error_mae:.3f})",
        )
        self._set_value_padding(ax_main, [true_values, pred_values])
        self._style_axes(
            ax_main,
            ylabel="Normalized Value",
            title="State Estimation vs Ground Truth",
        )
        ax_main.legend(frameon=True, loc="upper left")

        ax_res.axhline(0.0, color=self.colors["gray"], linewidth=1.0, linestyle=":")
        ax_res.bar(
            time_index,
            residual,
            width=0.9,
            color=self.colors["red"],
            alpha=0.55,
        )
        self._set_value_padding(ax_res, [residual], pad_ratio=0.15)
        self._style_axes(ax_res, xlabel="Time Step", ylabel="Error")

        save_path = os.path.join(self.base_dir, f"{name}_estimation.png")
        self._save_figure(fig, save_path, show=show)

    # ---------------- FULL COMPARISON ----------------
    def plot_full_comparison(self, x_true, y_obs, x_hat, name="full", num_points=200, show=False, jitter=False):
        clean_values = self._to_1d(x_true, num_points=num_points)
        corr_values = self._to_1d(y_obs, num_points=num_points)
        est_values = self._to_1d(x_hat, num_points=num_points)
        time_index = np.arange(len(clean_values))

        if jitter:
            value_range = float(np.max(clean_values) - np.min(clean_values)) if len(clean_values) else 0.0
            eps = max(value_range * 0.0025, 1e-4)
            corr_values = corr_values + eps
            est_values = est_values - eps

        fig, ax = self._new_figure(figsize=(11, 5.2))

        self._plot_line(
            ax,
            time_index,
            clean_values,
            label="Clean (x)",
            color=self.colors["blue"],
            linestyle="--",
            linewidth=2.3,
            alpha=0.95,
            zorder=4,
        )
        self._plot_line(
            ax,
            time_index,
            corr_values,
            label="Corrupted (y)",
            color=self.colors["red"],
            linestyle=":",
            linewidth=1.8,
            alpha=0.75,
            zorder=2,
        )
        self._plot_line(
            ax,
            time_index,
            est_values,
            label="Estimated (x̂)",
            color=self.colors["green"],
            linestyle="-",
            linewidth=2.1,
            alpha=0.92,
            zorder=3,
        )

        self._set_value_padding(ax, [clean_values, corr_values, est_values])
        self._style_axes(
            ax,
            xlabel="Time Step",
            ylabel="Normalized Value",
            title="Clean vs Corrupted vs Estimated",
        )
        ax.legend(frameon=True, loc="upper left")

        save_path = os.path.join(self.base_dir, f"{name}_full.png")
        self._save_figure(fig, save_path, show=show)
