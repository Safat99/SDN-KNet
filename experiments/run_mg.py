import yaml
import numpy as np
import tensorflow as tf
import argparse

from data.pipeline import TimeSeriesPipeline
from models.gru_baseline import GRUBaseline
from training.trainer import Trainer

from analysis.saver import ResultSaver
from analysis.plots import Plotter
from training.metrics import rmse, mae

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="gru")

    return parser.parse_args()

def main():
    
    args = parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override from SLURM
    config["corruption"]["delay"] = args.delay
    config["corruption"]["noise_std"] = args.noise
    
    
    # Set seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    

    # Pipeline
    pipeline = TimeSeriesPipeline(config)
    data, _, x_norm, y_norm = pipeline.run()

    X_train, Y_train = data["train"]
    X_val, Y_val = data["val"]
    X_test, Y_test = data["test"]

    # reshape for GRU: (batch, time, features)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Model
    model = GRUBaseline(
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        output_dim=config["window"]["forecast_horizon"]
    )

    # Trainer
    trainer = Trainer(model, config)

    # Train
    trainer.train(
        (X_train, Y_train),
        (X_val, Y_val)
    )
    
    # --- Predictions/TEST ---
    preds = model(X_test).numpy()
    
    # --- Metrics ---
    test_rmse = rmse(Y_test, preds).numpy()
    test_mae = mae(Y_test, preds).numpy()
    
    metrics = {
        "rmse": float(test_rmse),
        "mae": float(test_mae),
        "delay": args.delay,
        "noise": args.noise,
        "seed": args.seed
    }
        
    print("Test RMSE:", test_rmse)
    print("Test MAE:", test_mae)
    
    # ---------------- SAVE ----------------
    exp_name = f"gru_mg_d{args.delay}_n{args.noise}_s{args.seed}"
    
    saver = ResultSaver()
    saver.save_history(trainer.history, exp_name)
    saver.save_predictions(Y_test, preds, exp_name)
    saver.save_metrics(metrics, exp_name)
    
    # --- plot --- 
    plotter = Plotter()

    plotter.plot_training(trainer.history, name=exp_name)
    plotter.plot_predictions(Y_test, preds, name=exp_name)
    # plotter.plot_full_comparison(x_norm, y_norm, preds, name=exp_name)
    
    plotter.plot_full_comparison(
        x_norm[-len(preds):],
        y_norm[-len(preds):],
        preds.squeeze(),
        name=exp_name
    )


if __name__ == "__main__":
    main()