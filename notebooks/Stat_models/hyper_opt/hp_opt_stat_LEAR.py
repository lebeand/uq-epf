"""Optuna hyperparameter search for the LEAR (Lasso) baseline model."""

import sys  # noqa: INP001

sys.path.append("src/")

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn import linear_model

from model_training.data_modules.utils import EPFDataModule


def objective_function(trial, x_train, y_train, x_val, y_val, scale, max_iter):
    alpha = trial.suggest_float(
        "alpha", 1e-5, 1e-1, log=True
    )  # Lasso regularization parameter
    print(f"Trial {trial.number}: alpha = {alpha}")

    # train on train data and forecast val data
    reg_lasso = linear_model.Lasso(alpha=alpha, max_iter=max_iter)
    reg_lasso.fit(X=x_train, y=y_train)
    pred_val = reg_lasso.predict(X=x_val)

    val_loss = np.mean(np.abs((pred_val - y_val) * scale))
    print(f"Trial {trial.number}: Validation Loss = {val_loss:.3f}")

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

    seed = 0
    n_trials = 200
    max_iter = 1000

    sampler = TPESampler(seed=0)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=f"sqlite:///notebooks/Stat_models/LEAR_s{seed}_it{max_iter}.db",
    )
    study.optimize(
        lambda trial: objective_function(
            trial, x_train, y_train, x_val, y_val, scale, max_iter=max_iter
        ),
        n_trials=n_trials,
    )

    print("Best trial:")
    print(f"Value: {study.best_value}")
    print(f"Params: {study.best_params}")
