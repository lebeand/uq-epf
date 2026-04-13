"""Naive baseline model with residual-based Gaussian uncertainty estimates."""

import lightning as L
import numpy as np
import torch
from scipy.special import ndtri


class NaiveModel(L.LightningModule):
    def __init__(
        self,
        data_calibration,
        hparams_datamodule,
        input_processing=None,
        output_processing=None,
    ):
        super().__init__()

        self.save_hyperparameters()

        data_input_calibration = data_calibration[0]
        data_label_calibration = data_calibration[1]

        if self.hparams.output_processing is not None:
            offset, scale = self.hparams.output_processing
            data_label_calibration = data_label_calibration * scale + offset

        data_label_calibration = data_label_calibration.detach().numpy()
        output_calibration = self.inference(data_input_calibration).detach().numpy()
        error = np.abs(output_calibration - data_label_calibration)
        self.sigma = error.std(axis=0)

    def inference(self, x):
        if self.hparams.input_processing is not None:
            offset, scale = self.hparams.input_processing
            x = x * scale + offset

        # for monday, saturday, sunday take prices from week before
        output = []

        for i in range(x.shape[0]):
            if x[i, -1] == 1 or x[i, -2] == 1 or x[i, -7] == 1:
                output.append(x[i, 48 + 3 * 24 : 48 + 4 * 24])
            else:
                output.append(x[i, 48:72])
        output = torch.stack(output, dim=0)
        return output

    def ppf(self, quantile, x):
        output = self.inference(x).detach().numpy()
        return (
            ndtri(quantile)[:, None, None] * self.sigma[None, None, :]
            + output[None, :, :]
        )


if __name__ == "__main__":
    import sys

    sys.path.append("src/")
    import pandas as pd
    from lightning import seed_everything

    from model_training.data_modules.utils import EPFDataModule

    seed = 0
    standardization_case = "mean_std_input"
    start_date = None  # "2022-12-01"
    val_date = "2022-12-01"
    test_date = "2023-12-01"
    end_date = "2024-11-30"

    data_module = EPFDataModule(
        data_file_path="data/processed/smard_data_201810010000_202501010000.npz",
        val_date=val_date,
        test_date=test_date,
        end_date=end_date,
        start_date=start_date,
        batch_size=32,
        standardization_case=standardization_case,
    )

    val_input, val_labels = data_module.val_dataset[:]
    test_input, test_labels = data_module.test_dataset[:]
    test_labels = test_labels.detach().numpy()

    # Set the random seed for reproducibility
    seed_everything(seed)

    # Initialize the model
    model = NaiveModel(
        data_calibration=(val_input, val_labels),
        hparams_datamodule=data_module.hparams,
        input_processing=(
            data_module.offset_input,
            data_module.scale_input,
        ),
        output_processing=(
            data_module.offset_target,
            data_module.scale_target,
        ),
    )

    quantiles = np.linspace(0.01, 0.99, 99)
    significance_levels = np.flip(
        np.array([quantiles[-i - 1] - quantiles[i] for i in range(len(quantiles) // 2)])
    )
    data_input, data_labels = test_input, test_labels
    output = model.inference(data_input)
    output_q = model.ppf(quantiles, data_input)
