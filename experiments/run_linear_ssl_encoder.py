import yaml
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from data.pipeline import TimeSeriesPipeline
from models.gru_baseline import GRUBaseline
from training.trainer import Trainer
from analysis.saver import ResultSaver
from analysis.plots import Plotter
from utils import delay_estimation, noise_estimation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()



def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["corruption"]["delay"] = args.delay
    config["corruption"]["noise_std"] = args.noise

    # --------------- SEED ------------------
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # ---------------- PIPELINE ----------------
    pipeline = TimeSeriesPipeline(config)
    data, _, y_norm_full = pipeline.run_ssl()

    X_train, Y_train = data["train"]
    X_val, Y_val = data["val"]
    X_test, Y_test = data["test"]

    # dtype fix
    X_train = X_train.astype(np.float32)[..., np.newaxis]
    Y_train = Y_train.astype(np.float32)

    X_val = X_val.astype(np.float32)[..., np.newaxis]
    Y_val = Y_val.astype(np.float32)

    X_test = X_test.astype(np.float32)[..., np.newaxis]
    Y_test = Y_test.astype(np.float32)

    # ---------------- MODEL ----------------
    model = GRUBaseline(
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        output_dim=1
    )

    # ----------------- TRAIN -----------------------
    trainer = Trainer(model, config)
    trainer.train((X_train, Y_train), (X_val, Y_val))

    # ---------------- PREDICT y_hat ----------------
    y_hat = model(X_test, training=False).numpy().squeeze()
    y_true = Y_test.squeeze()

    # ---------------- RESIDUAL ----------------
    residual = y_true - y_hat

    print("Residual mean:", np.mean(residual))
    print("Residual variance:", np.var(residual))
    print("Residual shape:", residual.shape)
    
    d_hat, corrs = delay_estimation.estimate_delay(y_true, residual, max_delay=20)
    print(f"Estimated delay: {d_hat}")
    
    sigma2_hat = noise_estimation.estimate_noise_variance(residual)
    print(f"Estimated noise variance: {sigma2_hat}")
    
    # ---------------- SAVE ----------------
    # exp_name = f"ssl_gru_linear_d{args.delay}_n{args.noise}_s{args.seed}"

    # saver = ResultSaver("results/self-supervised-settings")
    # saver.save_history(trainer.history, exp_name)
    # saver.save_predictions(
    #     y_true,
    #     y_hat,
    #     X_test.squeeze()[:, -1],
    #     exp_name
    # )

    # metrics = {
    #     "residual_mean": float(np.mean(residual)),
    #     "residual_var": float(np.var(residual)),
    #     "delay": args.delay,
    #     "noise": args.noise,
    #     "seed": args.seed,
    #     "mode": "self_supervised_gru_encoder"
    # }

    # saver.save_metrics(metrics, exp_name)

    # # optional plot
    # plotter = Plotter("reports/figures/self-supervised-settings")
    # plotter.plot_training(trainer.history, name=exp_name)
    # plotter.plot_predictions(y_true, y_hat, name=exp_name)

    # # residual plot
    # fig_dir = "reports/figures"
    # os.makedirs(fig_dir, exist_ok=True)

    # plt.figure(figsize=(10, 4))
    # plt.plot(residual[:200])
    # plt.xlabel("Time Step")
    # plt.ylabel("Residual")
    # plt.title("Self-Supervised GRU Residual: y(t) - ŷ(t)")
    # plt.tight_layout()
    # plt.savefig(os.path.join(fig_dir, f"{exp_name}_residual.png"), dpi=300)
    # plt.close()


if __name__ == "__main__":
    main()