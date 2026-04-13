"""Optuna hyperparameter search for the XGBoost baseline model."""

import sys  # noqa: INP001

sys.path.append("src/")

import numpy as np
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBRegressor

from model_training.data_modules.utils import EPFDataModule


def objective_function(trial, x_train, y_train, x_val, y_val, scale):
    # Add hyperparameters for XGBoost
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
    subsample = trial.suggest_float("subsample", 0.6, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    gamma = trial.suggest_float("gamma", 0.0, 5.0)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 1.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.1, 10.0, log=True)

    print(f"\nTrial {trial.number}:")
    print(f"  n_estimators={n_estimators}, learning_rate={learning_rate:.4f}")
    print(f"  max_depth={max_depth}, min_child_weight={min_child_weight}")
    print(f"  subsample={subsample:.3f}, colsample_bytree={colsample_bytree:.3f}")
    print(
        f"  gamma={gamma:.3f}, reg_alpha={reg_alpha:.3f}, reg_lambda={reg_lambda:.3f}"
    )

    # Initialize XGBoost regressor
    reg_xgb = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        n_jobs=-1,  # Use all available cores
        verbosity=0,  # Suppress XGBoost warnings
    )

    # Train on train data and predict on val data
    reg_xgb.fit(X=x_train, y=y_train)
    pred_val = reg_xgb.predict(X=x_val)

    # Calculate MAE on validation set
    val_loss = np.mean(np.abs((pred_val - y_val) * scale))
    print(f"  Validation MAE = {val_loss:.3f}")

    return val_loss


if __name__ == "__main__":
    # load data
    data_module = EPFDataModule(
        data_file_path="data/processed/smard_data_201810010000_202501010000.npz",
        val_date="2022-12-01",
        test_date="2023-12-01",
        end_date="2024-11-30",
        batch_size=32,
        standardization_case="mean_std",
    )

    x_train, y_train = data_module.train_dataset[:]
    x_val, y_val = data_module.val_dataset[:]

    # Convert tensors to numpy arrays
    x_train = x_train.numpy()
    y_train = y_train.numpy()
    x_val = x_val.numpy()
    y_val = y_val.numpy()

    # Get scale and offset for denormalization
    offset = data_module.offset_target.numpy()
    scale = data_module.scale_target.numpy()

    seed = 42
    n_trials = 200

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=f"sqlite:///notebooks/Stat_models/hyper_opt/XGBoost_s{seed}.db",
        study_name=f"XGBoost_optimization_seed{seed}",
        load_if_exists=True,  # Resume if study exists
    )

    print(f"Starting XGBoost hyperparameter optimization with {n_trials} trials...")
    print(f"Database: notebooks/Stat_models/hyper_opt/XGBoost_s{seed}.db")

    study.optimize(
        lambda trial: objective_function(trial, x_train, y_train, x_val, y_val, scale),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best validation MAE: {study.best_value:.3f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
