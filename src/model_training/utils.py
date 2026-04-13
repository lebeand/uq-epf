"""Utility helpers for running Lightning model training workflows."""

import sys

sys.path.append("src/")

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger


def train_model(
    model, data_module, max_epochs=2, log_dir="logs_test", name="Test", callbacks=None
):
    if callbacks is None:
        callbacks = []

    # Set up the TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=name,
        log_graph=True,
        default_hp_metric=False,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss_early_stopping",
        min_delta=0.0,
        patience=100,
        verbose=False,
        mode="min",
    )
    callbacks.append(early_stop_callback)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss_early_stopping",
        mode="min",
    )
    callbacks.append(checkpoint_callback)

    # Initialize the trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        deterministic=True,
        log_every_n_steps=0,
        enable_progress_bar=False,
        callbacks=callbacks,
    )

    # Train and test the model (start tensorboard with `tensorboard --logdir logs/DeepNeuralNetwork_test`)
    trainer.fit(model, data_module)
    # trainer.test(model, data_module)

    return trainer


if __name__ == "__main__":
    pass
