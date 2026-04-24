import optuna
import yaml
import numpy as np
import tensorflow as tf

from data.pipeline import TimeSeriesPipeline
from models.gru_baseline import GRUBaseline
from training.trainer import Trainer
from training.metrics import rmse
from analysis.saver import ResultSaver

def objective(trial):

    # ---------------- SEARCH SPACE ----------------
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    input_length = trial.suggest_categorical("input_length", [10, 20, 30])

    # ---------------- LOAD CONFIG ----------------
    with open("configs/linear.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["model"]["hidden_size"] = hidden_size
    config["model"]["num_layers"] = num_layers
    config["training"]["learning_rate"] = lr
    config["training"]["batch_size"] = batch_size
    config["window"]["input_length"] = input_length

    # Fix corruption for tuning (VERY IMPORTANT)
    config["corruption"]["delay"] = 0
    config["corruption"]["noise_std"] = 0.0

    # Seed for reproducibility
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # ---------------- DATA ----------------
    pipeline = TimeSeriesPipeline(config)
    data, _, _, _ = pipeline.run()

    X_train, Y_train = data["train"]
    X_val, Y_val = data["val"]

    X_train = X_train[..., np.newaxis].astype(np.float32)
    X_val = X_val[..., np.newaxis].astype(np.float32)

    Y_train = Y_train.astype(np.float32)
    Y_val = Y_val.astype(np.float32)

    # ---------------- MODEL ----------------
    model = GRUBaseline(
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=1
    )

    trainer = Trainer(model, config)
    trainer.train((X_train, Y_train), (X_val, Y_val))

    preds = model(X_val).numpy()

    val_rmse = rmse(Y_val, preds).numpy()
    
    # -------- SAVE TRIAL RESULT --------
    saver = ResultSaver(base_dir="results/tuning_linear")

    trial_result = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "learning_rate": lr,
        "batch_size": batch_size,
        "input_length": input_length,
        "val_rmse": float(val_rmse)
    }

    saver.save_metrics(trial_result, f"trial_{trial.number}")

    return val_rmse


if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(objective, n_trials=30)

    
    # -------- SAVE BEST PARAMS --------
    saver = ResultSaver(base_dir="results/tuning_linear")

    best_result = {
        "best_params": study.best_params,
        "best_rmse": float(study.best_value)
    }
    
    saver.save_metrics(best_result, "best_params")
    
    print("Best params:", study.best_params)
    print("Best RMSE:", study.best_value)
