"""Lightning data modules and preprocessing utilities for EPF experiments."""

import os

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


# preprocessing module
class DataProcessingModule(L.LightningModule):
    def __init__(self, standardization_case, X_train, y_train=None):
        super().__init__()
        self.standardization_case = standardization_case

        self.offset_input = torch.tensor(0, requires_grad=False)
        self.scale_input = torch.tensor(1, requires_grad=False)
        self.offset_target = torch.tensor(0, requires_grad=False)
        self.scale_target = torch.tensor(1, requires_grad=False)

        if standardization_case in ("mean_std_input", "mean_std"):
            # Standardize input data except weekday dummies
            standardization_mean_input = X_train[:, :-7].mean(dim=0)
            standardization_std_input = X_train[:, :-7].std(dim=0)

            self.offset_input = torch.cat(
                [standardization_mean_input, torch.zeros(7)], dim=0
            )
            self.scale_input = torch.cat(
                [standardization_std_input, torch.ones(7)], dim=0
            )
        if standardization_case == "mean_std":
            # Standardize output data
            standardization_mean_target = y_train.mean(dim=0)
            standardization_std_target = y_train.std(dim=0)

            self.offset_target = standardization_mean_target
            self.scale_target = standardization_std_target

        if standardization_case in ("median_mad_adj_input", "median_mad_adj"):
            # Standardize input and output data with  median and median absolut deviation except weekday dummies
            standardization_median_input = X_train[:, :-7].median(dim=0)[0]
            standardization_mad_adj_input = (
                X_train[:, :-7] - standardization_median_input
            ).abs().median(dim=0)[0] * 1.4826

            self.offset_input = torch.cat(
                [standardization_median_input, torch.zeros(7)], dim=0
            )
            self.scale_input = torch.cat(
                [standardization_mad_adj_input, torch.ones(7)], dim=0
            )

        if standardization_case == "median_mad_adj":
            standardization_median_target = y_train.median(dim=0)[0]
            standardization_mad_adj_target = (
                y_train - standardization_median_target
            ).abs().median(dim=0)[0] * 1.4826

            self.offset_target = standardization_median_target
            self.scale_target = standardization_mad_adj_target

        if standardization_case in ("min_max_input", "min_max"):
            # Standardize input data except weekday dummies
            standardization_min_input = X_train[:, :-7].min(dim=0)[0]
            standardization_max_input = X_train[:, :-7].max(dim=0)[0]

            self.offset_input = torch.cat(
                [standardization_min_input, torch.zeros(7)], dim=0
            )
            self.scale_input = torch.cat(
                [standardization_max_input - standardization_min_input, torch.ones(7)],
                dim=0,
            )
        if standardization_case == "min_max":
            # Standardize input and output data except weekday dummies
            standardization_min_target = y_train.min(dim=0)[0]
            standardization_max_target = y_train.max(dim=0)[0]

            self.offset_target = standardization_min_target
            self.scale_target = standardization_max_target - standardization_min_target

    def preprocess_input(self, x):
        n_unsqueeze = x.ndim - self.offset_input.ndim
        return (x - self.offset_input[(None,) * n_unsqueeze]) / self.scale_input[
            (None,) * n_unsqueeze
        ]

    def preprocess_target(self, y):
        n_unsqueeze = y.ndim - self.offset_target.ndim
        return (y - self.offset_target[(None,) * n_unsqueeze]) / self.scale_target[
            (None,) * n_unsqueeze
        ]

    def postprocess_input(self, x):
        n_unsqueeze = x.ndim - self.offset_input.ndim
        return (
            x * self.scale_input[(None,) * n_unsqueeze]
            + self.offset_input[(None,) * n_unsqueeze]
        )

    def postprocess_target(self, y):
        n_unsqueeze = y.ndim - self.offset_target.ndim
        return (
            y * self.scale_target[(None,) * n_unsqueeze]
            + self.offset_target[(None,) * n_unsqueeze]
        )

    def postprocess_gauss(self, mu, sigma):
        n_unsqueeze = mu.ndim - self.offset_target.ndim
        return mu * self.scale_target[(None,) * n_unsqueeze] + self.offset_target[
            (None,) * n_unsqueeze
        ], sigma * self.scale_target[(None,) * n_unsqueeze]


class EPFDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_file_path,
        val_date: str,
        test_date: str,
        batch_size: int = 32,
        start_date=None,
        end_date=None,
        standardization_case=None,
    ):
        super().__init__()

        # Validate input parameters
        assert isinstance(data_file_path, str), "data_file_path must be a string"
        assert isinstance(batch_size, int), "batch_size must be an integer"
        assert isinstance(val_date, str), "val_date must be a string"
        assert isinstance(test_date, str), "test_date must be a string"
        if start_date is not None:
            assert isinstance(start_date, str), "start_date must be a string"
        if end_date is not None:
            assert isinstance(end_date, str), "end_date must be a string"

        if standardization_case is not None:
            if standardization_case not in (
                "mean_std_input",
                "mean_std",
                "median_mad_adj_input",
                "median_mad_adj",
                "min_max_input",
                "min_max",
            ):
                raise ValueError(
                    "standardization_case must be one of 'mean_std_input', 'mean_std', 'median_mad_adj_input', 'median_mad_adj', 'min_max_input', 'min_max'"
                )
        else:
            standardization_case = ""

        # Save hyperparameters
        self.save_hyperparameters()

        self.batch_size = batch_size

        val_date = pd.Timestamp(val_date)
        test_date = pd.Timestamp(test_date)
        if start_date is not None:
            start_date = pd.Timestamp(start_date)
        if end_date is not None:
            end_date = pd.Timestamp(end_date)

        # Load data
        data = np.load(data_file_path)
        X = torch.tensor(data["features"], dtype=torch.float32)
        y = torch.tensor(data["targets"], dtype=torch.float32)
        dates = data["dates"]

        # Split data into train, val, test sets
        start_date_idx = (
            np.where(dates == start_date)[0][0] if start_date is not None else 0
        )
        val_date_idx = np.where(dates == val_date)[0][0]
        test_date_idx = np.where(dates == test_date)[0][0]
        end_date_idx = (
            np.where(dates == end_date)[0][0] + 1 if end_date is not None else -1
        )
        X_train, X_val, X_test = (
            X[start_date_idx:val_date_idx],
            X[val_date_idx:test_date_idx],
            X[test_date_idx:end_date_idx],
        )
        y_train, y_val, y_test = (
            y[start_date_idx:val_date_idx],
            y[val_date_idx:test_date_idx],
            y[test_date_idx:end_date_idx],
        )

        # Create data processing module
        self.data_processing_module = DataProcessingModule(
            standardization_case, X_train, y_train
        )

        # Preprocess data
        X_train = self.data_processing_module.preprocess_input(X_train)
        y_train = self.data_processing_module.preprocess_target(y_train)
        X_val = self.data_processing_module.preprocess_input(X_val)
        y_val = self.data_processing_module.preprocess_target(y_val)
        X_test = self.data_processing_module.preprocess_input(X_test)
        y_test = self.data_processing_module.preprocess_target(y_test)

        # compatibility with older version
        self.offset_input = self.data_processing_module.offset_input
        self.scale_input = self.data_processing_module.scale_input
        self.offset_target = self.data_processing_module.offset_target
        self.scale_target = self.data_processing_module.scale_target

        # Create datasets
        self.train_dataset = TensorDataset(X_train, y_train)
        self.val_dataset = TensorDataset(X_val, y_val)
        self.test_dataset = TensorDataset(X_test, y_test)

        # Save dimensions and number of samples
        self.feature_dim = X.shape[1]
        self.target_dim = y.shape[1]
        self.total_samples = X.shape[0]
        self.train_samples = X_train.shape[0]
        self.val_samples = X_val.shape[0]
        self.test_samples = X_test.shape[0]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=int(np.min([os.cpu_count() - 1, 7])),
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=int(np.min([os.cpu_count() - 1, 7])),
            persistent_workers=True,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )


## Example usage
if __name__ == "__main__":
    val_date = "2022-12-01"
    test_date = "2023-12-01"
    end_date = "2024-11-30"
    data_file_path = "../data/processed/smard_data_2024-12-11.npz"
    standardization_case = "mean_std"

    data_module = EPFDataModule(
        data_file_path=data_file_path,
        val_date=val_date,
        test_date=test_date,
        end_date=end_date,
        batch_size=32,
        standardization_case=standardization_case,
    )

    train_input, train_labels = data_module.train_dataset[:]
    val_input, val_labels = data_module.val_dataset[:]
    test_input, test_labels = data_module.test_dataset[:]

    data_input, data_labels = test_input, test_labels
    # Postprocess data
    data_labels = data_labels * data_module.scale_target + data_module.offset_target
    # or data_labels = data_module.data_processing_module.postprocess_target(data_labels)
    data_labels = data_labels.detach().numpy()
