"""Evidential neural network for probabilistic day-ahead price forecasting."""

import lightning as L
import numpy as np
import torch
from scipy import stats
from torch import nn


def loss_evidenial(y, gamma, lognu, logalpham1, logbeta, evidence_regularizer=None):
    # ensure tolerance nu > 1e-6, alpha > 1+1e-6, beta > 1e-6
    tol = np.log(1e-6)
    lognu = torch.clamp(lognu, min=tol)
    logalpham1 = torch.clamp(logalpham1, min=tol)
    logbeta = torch.clamp(logbeta, min=tol)

    alpha = logalpham1.exp() + 1
    nu = lognu.exp()

    logomega = torch.tensor(2).log() + logbeta + torch.log(1 + nu)
    omega = logomega.exp()

    loss = torch.mean(
        -0.5 * lognu
        - alpha * logomega
        + (alpha + 0.5) * torch.log((y - gamma) ** 2 * nu + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    if evidence_regularizer is not None:
        loss += torch.mean(
            evidence_regularizer * torch.abs(y - gamma) * (2 * nu + alpha)
        )

    return loss


class EvidentialDeepNeuralNetworkNormal(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_batch_norm,
        hidden_dims=None,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.0,
        l2_regularization: float = 0.0,
        evidence_regularizer: float = 0.0,
        hparams_datamodule=None,
        output_postprocessing=None,
    ):
        super().__init__()

        # Set default value
        if hidden_dims is None:
            hidden_dims = []

        # Validate input parameters
        if not isinstance(input_dim, int):
            error_message = "input_dim must be an integer"
            raise TypeError(error_message)
        assert isinstance(output_dim, int), "output_dim must be an integer"
        assert isinstance(hidden_dims, list), "hidden_dim must be a list"
        if len(hidden_dims) > 0:
            assert all(isinstance(dim, int) for dim in hidden_dims), (
                "hidden_dim must be a list of integers"
            )
        assert isinstance(learning_rate, float), "learning_rate must be a float"
        assert isinstance(dropout_rate, float), "dropout_rate must be a float"
        assert isinstance(l2_regularization, float), "l2_regularization must be a float"
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

        # # Initialize an example input tensor for TensorBoard graph visualization
        # self.example_input_array = torch.zeros(1, input_dim)

        # Define the network architecture
        if len(hidden_dims) == 0:
            self.output_layer = nn.Linear(input_dim, 4 * output_dim)
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
            if dropout_rate > 0:
                self.hidden_layers[0].append(nn.Dropout(dropout_rate))

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
                if dropout_rate > 0:
                    self.hidden_layers[i].append(nn.Dropout(dropout_rate))

            # Add output layer
            self.output_layer = nn.Linear(hidden_dims[-1], 4 * output_dim)
            self.initialize_weights_xavier_(self.output_layer)

        # self.softplus_fn = nn.Softplus()

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

    def forward(self, x):
        if hasattr(self, "hidden_layers"):
            x = self.hidden_layers(x)
        x = self.output_layer(x)
        return (
            x[:, 0 : self.hparams.output_dim],
            x[:, self.hparams.output_dim : self.hparams.output_dim * 2],
            x[:, self.hparams.output_dim * 2 : self.hparams.output_dim * 3],
            x[:, self.hparams.output_dim * 3 :],
        )

    @torch.no_grad()
    def inference(self, x):
        output, _, _, _ = self.forward(x)
        if self.hparams.output_postprocessing is not None:
            offset, scale = self.hparams.output_postprocessing
            output = output * scale + offset
        return output.double()

    @torch.no_grad()
    def ppf(self, quantile, x):
        gamma, lognu, logalpham1, logbeta = self.forward(x)

        nu = lognu.exp()
        alpha = logalpham1.exp() + 1
        beta = logbeta.exp()

        if self.hparams.output_postprocessing is not None:
            offset, scale = self.hparams.output_postprocessing
            gamma = gamma * scale + offset
            beta = beta * scale

        gamma = gamma.detach().numpy().astype(np.float64)
        nu = nu.detach().numpy().astype(np.float64)
        alpha = alpha.detach().numpy().astype(np.float64)
        beta = beta.detach().numpy().astype(np.float64)

        distr_student_t = stats.t(
            df=2 * alpha, loc=gamma, scale=beta * (1 + nu) / nu / alpha
        )

        return np.stack([distr_student_t.ppf(q) for q in quantile])

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams,
            {"hp_val_loss": 500.0},  # set to a high value
        )  # could also log only val_loss but the first step has two values then

    def training_step(self, batch, batch_idx):
        x, y = batch
        gamma, lognu, logalpham1, logbeta = self.forward(x)

        loss = loss_evidenial(
            y,
            gamma,
            lognu,
            logalpham1,
            logbeta,
            self.hparams.evidence_regularizer,
        )

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
        gamma, lognu, logalpham1, logbeta = self.forward(x)
        loss = loss_evidenial(
            y,
            gamma,
            lognu,
            logalpham1,
            logbeta,
        )


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
        gamma, lognu, logalpham1, logbeta = self.forward(x)
        loss = loss_evidenial(
            y,
            gamma,
            lognu,
            logalpham1,
            logbeta,
        )
        
        self.log("loss/test", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.l2_regularization,
        )


if __name__ == "__main__":
    import sys

    sys.path.append("src/")
    from lightning import seed_everything

    from model_training.data_modules.utils import EPFDataModule
    from model_training.utils import train_model

    n_runs = 10
    max_epochs = 2000
    seed = 0
    log_dir = "logs_test/"

    standardization_case = "mean_std"
    name = f"EvDNN_Normal_{standardization_case}_hp-right_nll"

    start_date = None  # "2022-12-01"
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
        model = EvidentialDeepNeuralNetworkNormal(
            input_dim=data_module.feature_dim,
            output_dim=data_module.target_dim,
            hidden_dims=hidden_dims,
            learning_rate=1e-5,
            hparams_datamodule=data_module.hparams,
            use_batch_norm=False,
            l2_regularization=1e-3,
            evidence_regularizer=1e-2,
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
