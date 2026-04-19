import yaml
import numpy as np
import tensorflow as tf

from data.pipeline import TimeSeriesPipeline
from models.gru_baseline import GRUBaseline
from training.trainer import Trainer

from analysis.saver import ResultSaver
from analysis.plots import Plotter
from training.metrics import rmse, mae

def main():
    # Load config
    with open("configs/mackey_glass.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Pipeline
    pipeline = TimeSeriesPipeline(config)
    data, _ = pipeline.run()

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
    
    # --- Predictions ---
    preds = model(X_test).numpy()
    
    # --- Metrics ---
    test_rmse = rmse(Y_test, preds).numpy()
    test_mae = mae(Y_test, preds).numpy()
    
    metrics = {
    "rmse": float(test_rmse),
    "mae": float(test_mae)
    }

    print("Test RMSE:", test_rmse)
    print("Test MAE:", test_mae)
    
    saver = ResultSaver()

    saver.save_history(trainer.history, "gru_mg")
    saver.save_predictions(Y_test, preds, "gru_mg")
    saver.save_metrics(metrics, "gru_mg")
    
    # --- plot --- 
    plotter = Plotter()

    plotter.plot_training(trainer.history, name="gru_mg")
    plotter.plot_predictions(Y_test, preds, name="gru_mg")


if __name__ == "__main__":
    main()