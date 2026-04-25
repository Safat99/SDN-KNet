import yaml
import numpy as np
import tensorflow as tf
import argparse

from data.pipeline import TimeSeriesPipeline
from models.gru_baseline import GRUBaseline
from models.kf_baselines import run_kf_linear
from training.trainer import Trainer
from training.metrics import rmse, mae
from analysis.saver import ResultSaver
from analysis.plots import Plotter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, default="gru")
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_em", action="store_true") # to handle the 'use_em' case of KF

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    if dataset_name != "linear":
        raise ValueError(
            f"run_linear.py expects dataset.name='linear', got '{dataset_name}' from {args.config}"
        )

    # Override from SLURM
    config["corruption"]["delay"] = args.delay
    config["corruption"]["noise_std"] = args.noise

    # ---- seed ----
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # ---- pipeline -----
    pipeline = TimeSeriesPipeline(config)
    data, _, x_norm, y_norm = pipeline.run()

    X_train, Y_train = data["train"]
    X_val, Y_val = data["val"]
    X_test, Y_test = data["test"]
    
    # cast to float32 for tf as numpy default is float64    
    X_train = X_train.astype(np.float32)
    Y_train = Y_train.astype(np.float32)

    X_val = X_val.astype(np.float32)
    Y_val = Y_val.astype(np.float32)

    X_test = X_test.astype(np.float32)
    Y_test = Y_test.astype(np.float32)
    
    # reshape for gru
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # ---------------- MODEL ----------------
    if args.model == "gru":
        model = GRUBaseline(
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            output_dim=1
        )

        trainer = Trainer(model, config)
        trainer.train((X_train, Y_train), (X_val, Y_val))

        preds = model(X_test).numpy()

    elif args.model == "kf":
        preds_full = run_kf_linear(
            y_norm, 
            config,
            use_em=args.use_em
        )
        preds = preds_full[-len(Y_test):]
        preds = preds.reshape(Y_test.shape)
        trainer = None

    # ---------------- METRICS ----------------
    test_rmse = rmse(Y_test, preds).numpy()
    test_mae = mae(Y_test, preds).numpy()
    
    metrics = {
        "rmse": float(test_rmse),
        "mae": float(test_mae),
        "delay": args.delay,
        "noise": args.noise,
        "seed": args.seed,
        "model": args.model,
        "use_em": args.use_em if args.model == "kf" else None,
        
        # for debugging anomalies
        "n_test": len(Y_test),
        "mean_pred": float(np.mean(preds)),
        "std_pred": float(np.std(preds)),
        "mean_true": float(np.mean(Y_test)),
        "std_true": float(np.std(Y_test))
    }

    print("RMSE:", test_rmse)
    print("MAE:", test_mae)

    # ---------------- SAVE ----------------
    # exp_name = f"{args.model}_linear_d{args.delay}_n{args.noise}_s{args.seed}"
    exp_name = f"{args.model}_linear_d{args.delay}_n{args.noise}_s{args.seed}_em{int(args.use_em)}"
    
    if args.model == "kf" and args.use_em:
        exp_name += "_with_em"

    saver = ResultSaver()
    saver.save_predictions(x_true=Y_test, x_hat=preds, y_observed=y_norm[-len(preds):], name=exp_name)
    saver.save_metrics(metrics, exp_name)

    plotter = Plotter()

    if trainer is not None:
        saver.save_history(trainer.history, exp_name)
        plotter.plot_training(trainer.history, name=exp_name)

    plotter.plot_predictions(Y_test, preds, name=exp_name)

    plotter.plot_full_comparison(
        x_norm[-len(preds):],
        y_norm[-len(preds):],
        preds.squeeze(),
        name=exp_name,
        jitter=True
    )


if __name__ == "__main__":
    main()