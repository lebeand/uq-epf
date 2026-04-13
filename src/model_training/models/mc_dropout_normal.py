"""Monte Carlo Dropout neural network with Gaussian predictive outputs."""

import sys

sys.path.append("src/")

import lightning as L
import numpy as np
import torch
from torch import nn

from model_training.models.utils import ppf_gm


def loss_normal_nll(y, y_hat, logs2):
    # ensure tolerance sigma**2 > 1e-6
    tol = np.log(1e-6)
    logs2 = torch.clamp(logs2, min=tol)

    return 0.5 * torch.mean(logs2 + torch.exp(-logs2) * (y_hat - y) ** 2)


def nll_gm_loss(y, y_hat, logs2):
    # ensure tolerance sigma**2 > 1e-6
    tol = np.log(1e-6)
    logs2 = torch.clamp(logs2, min=tol)

    # log-sum-exp trick for numerical stability
    temp = -0.5 * (torch.exp(-logs2) * (y - y_hat) ** 2 + logs2)
    logsumexp_trick = torch.max(temp, dim=0)[0]

    loss = -torch.mean(
        logsumexp_trick
        + torch.log(
            torch.mean(
                torch.exp(temp - logsumexp_trick),
                dim=0,
            )
        )
    )
    return loss


class MCDropoutNormal(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_batch_norm,
        hidden_dims=None,
        learning_rate: float = 1e-3,
        dropout_rate=None,
        l2_regularization: float = 0.0,
        forward_passes: int = 10,
        hparams_datamodule=None,
        output_postprocessing=None,
    ):
        super().__init__()

        # Set default value
        if hidden_dims is None:
            hidden_dims = []
        if dropout_rate is None:
            dropout_rate = [0.0]

        # Validate input parameters
        if not isinstance(input_dim, int):
            error_message = "input_dim must be an integer"
            raise TypeError(error_message)
        assert isinstance(output_dim, int), "output_dim must be an integer"
        assert isinstance(hidden_dims, list), "hidden_dim must be a list"
        if len(hidden_dims) > 0:
            assert all(
                isinstance(dim, int) for dim in hidden_dims
            ), "hidden_dim must be a list of integers"
        assert isinstance(learning_rate, float), "learning_rate must be a float"
        assert isinstance(dropout_rate, list), "dropout_rate must be a list"
        assert (
            len(dropout_rate) == len(hidden_dims) or len(dropout_rate) == 1
        ), "dropout_rate must be a list of length len(hidden_dims)+1 or a single value"
        assert all(
            isinstance(rate, float) for rate in dropout_rate
        ), "dropout_rate must be a list of floats"
        assert isinstance(l2_regularization, float), "l2_regularization must be a float"
        assert isinstance(forward_passes, int), "forward_passes must be an integer"
        assert isinstance(use_batch_norm, bool), "batch_norm_bool must be a boolean"
        if hparams_datamodule is not None:
            assert isinstance(
                hparams_datamodule,
                dict,
            ), "hparams_datamodule must be a dictionary"

        if output_postprocessing is None:
            output_postprocessing = (0.0, 1.0)

        # Save hyperparameters for checkpointing
        self.save_hyperparameters()

        # Initialize an example input tensor for TensorBoard graph visualization
        self.example_input_array = torch.zeros(1, input_dim)

        # Define the network architecture
        if len(hidden_dims) == 0:
            self.output_layer = nn.Linear(input_dim, 2 * output_dim)
            self.initialize_weights_xavier_(self.output_layer)
        else:
            self.hidden_layers = nn.Sequential()
            # Add first hidden layer
            self.hidden_layers.append(
                nn.Sequential(nn.Linear(input_dim, hidden_dims[0])),
            )
            # Initialization
            self.initialize_weights_kaiming_(self.hidden_layers[0][0])
            # batch norm
            if use_batch_norm:
                self.hidden_layers[0].append(nn.BatchNorm1d(hidden_dims[0]))
            # activation function
            self.hidden_layers[0].append(nn.ReLU())
            # dropout
            if dropout_rate[0] > 0:
                self.hidden_layers[0].append(nn.Dropout(dropout_rate[0]))

            # Add remaining hidden layers
            for i in range(1, len(hidden_dims)):
                self.hidden_layers.append(
                    nn.Sequential(nn.Linear(hidden_dims[i - 1], hidden_dims[i])),
                )
                # Initialization
                self.initialize_weights_kaiming_(self.hidden_layers[i][0])
                # batch norm
                if use_batch_norm:
                    self.hidden_layers[i].append(nn.BatchNorm1d(hidden_dims[i]))
                # activation function
                self.hidden_layers[i].append(nn.ReLU())
                # dropout
                if len(dropout_rate) > 1:
                    if dropout_rate[i] > 0:
                        self.hidden_layers[i].append(nn.Dropout(dropout_rate[i]))
                elif dropout_rate[0] > 0:
                    self.hidden_layers[i].append(nn.Dropout(dropout_rate[0]))

            # Add output layer
            self.output_layer = nn.Linear(hidden_dims[-1], 2 * output_dim)
            self.initialize_weights_xavier_(self.output_layer)

        # Initialize lists to store outputs
        self.train_step_outputs = []
        self.validation_step_outputs = []

        # Initialize validation loss for hyperparameter metric
        self.hp_val_loss = 500.0  # set to a high value

    def initialize_weights_kaiming_(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def initialize_weights_xavier_(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def train(self, mode: bool = True):  # noqa: FBT001, FBT002
        super().train(mode)

        # Dropout layers always in train mode
        if hasattr(self, "hidden_layers") and not mode:

            def enable_dropout(mod: nn.Module) -> None:
                if isinstance(mod, nn.Dropout):
                    mod.train(mode=True)

            self.apply(enable_dropout)

        return self

    def forward(self, x):
        if hasattr(self, "hidden_layers"):
            x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x[:, : self.hparams.output_dim], x[:, self.hparams.output_dim :]

    @torch.no_grad()
    def inference(self, x, forward_passes=None):
        if forward_passes is None:
            forward_passes = self.hparams.forward_passes

        output = [self.forward(x) for _ in range(forward_passes)]
        output, _ = zip(*output, strict=False)
        output = torch.stack(output).double().mean(dim=0)
        if self.hparams.output_postprocessing is not None:
            offset, scale = self.hparams.output_postprocessing
            output = output * scale + offset

        return output

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams,
            {"hp_val_loss": 500.0},  # set to a high value
        )  # could also log only val_loss but the first step has two values then

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, logs2 = self.forward(x)
        loss = loss_normal_nll(y, y_hat, logs2)

        if self.trainer.state.stage == "train" and self.trainer.state.fn == "fit":
            self.train_step_outputs.append(loss)
        return loss

    @torch.no_grad()
    def on_train_epoch_end(self):
        if self.trainer.state.stage == "train" and self.trainer.state.fn == "fit":
            avg_train_loss = torch.stack(self.train_step_outputs).mean()
            self.logger.log_metrics(
                {"loss/train": avg_train_loss},
                step=self.current_epoch,
            )
        self.train_step_outputs.clear()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = [self.forward(x) for _ in range(self.hparams.forward_passes)]
        y_hat, logs2 = zip(*output, strict=False)
        y_hat = torch.stack(y_hat)
        logs2 = torch.stack(logs2)

        loss = nll_gm_loss(y, y_hat, logs2)

        if self.trainer.state.stage == "validate" and self.trainer.state.fn == "fit":
            self.validation_step_outputs.append(loss)
        return loss

    @torch.no_grad()
    def on_validation_epoch_end(self):
        if self.trainer.state.stage == "validate" and self.trainer.state.fn == "fit":
            avg_val_loss = torch.stack(self.validation_step_outputs).mean()
            self.log(
                "val_loss_early_stopping",
                avg_val_loss,
                logger=False,
            )  # only for early stopping callback
            self.logger.log_metrics(
                {"loss/val": avg_val_loss},
                step=self.current_epoch,
            )
            if avg_val_loss < self.hp_val_loss:
                self.hp_val_loss = avg_val_loss
                self.log(
                    "val_loss_neptuna",
                    avg_val_loss,
                    logger=False,
                )  # only for neptuna
                self.logger.log_metrics({"hp_val_loss": avg_val_loss})
        self.validation_step_outputs.clear()

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x, y = batch
        output = [self.forward(x) for _ in range(self.hparams.forward_passes)]
        y_hat, logs2 = zip(*output, strict=False)
        y_hat = torch.stack(y_hat)
        logs2 = torch.stack(logs2)

        loss = nll_gm_loss(y, y_hat, logs2)

        self.log("loss/test", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.l2_regularization,
        )

    @torch.no_grad()
    def ppf(self, quantile, x, xtol=1e-5):
        output = [self.forward(x) for _ in range(self.hparams.forward_passes)]
        mu, logs2 = zip(*output, strict=False)
        mu = torch.stack(mu)
        sigma = torch.stack(logs2).exp().sqrt()

        offset = self.hparams.output_postprocessing[0][None, None, :]
        scale = self.hparams.output_postprocessing[1][None, None, :]

        mu = mu * scale + offset
        sigma = sigma * scale

        mu = mu.detach().numpy().astype(np.float64)
        sigma = sigma.detach().numpy().astype(np.float64)

        return ppf_gm(mu, sigma, quantile, xtol)


if __name__ == "__main__":
    import sys

    sys.path.append("src/")
    from lightning import seed_everything

    from model_training.data_modules.utils import EPFDataModule
    from model_training.utils import train_model

    n_runs = 1
    max_epochs = 2
    seed = 0
    log_dir = "logs_test/"
    forward_passes = 10

    standardization_case = "mean_std"
    name = f"MCDropout{forward_passes}Normal_{standardization_case}_hp"

    start_date = None  
    val_date = "2022-12-01"
    test_date = "2023-12-01"
    end_date = "2024-11-30"
    hidden_dims = [1024, 1024]

    data_module = EPFDataModule(
        data_file_path="data/processed/smard_data_2024-12-11.npz",
        val_date=val_date,
        test_date=test_date,
        end_date=end_date,
        start_date=start_date,
        batch_size=32,
        standardization_case=standardization_case,
    )

    for _ in range(n_runs):
        # Set the random seed for reproducibility
        seed_everything(seed)

        # Initialize the model
        model = MCDropoutNormal(
            input_dim=data_module.feature_dim,
            output_dim=data_module.target_dim,
            hidden_dims=hidden_dims,
            learning_rate=1e-4,
            dropout_rate=[0.1],
            l2_regularization=1e-4,
            hparams_datamodule=data_module.hparams,
            use_batch_norm=False,
            forward_passes=forward_passes,
            output_postprocessing=(
                data_module.offset_target,
                data_module.scale_target,
            ),
        )

        # Train the model
        train_model(
            model,
            data_module,
            max_epochs=max_epochs,
            log_dir=log_dir,
            name=name,
        )

        seed = seed + 1
