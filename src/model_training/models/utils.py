"""Utilities for combining distributional neural network model outputs."""

import os
import sys

import lightning as L
import numpy as np
import scipy.optimize as opt
import torch
from joblib import Parallel, delayed
from scipy.special import ndtr, ndtri

sys.path.append("src/")


from model_training.models.ddnn_normal import DistributionalDeepNeuralNetworkNormal


class EnsembleDDNNNormal(L.LightningModule):
    def __init__(
        self,
        path=None,
        models_checkpoints=None,
        n_models: int = 1,
        version_offset: int = 0,
        combination_approach=None,
    ):
        super().__init__()

        if combination_approach is None:
            combination_approach = "gm"
        if combination_approach not in ("gm", "avq"):
            raise ValueError(f"Unknown combination approach: {combination_approach}")

        self.save_hyperparameters()

        self.models = []

        if path is not None:
            for i in range(n_models):
                path_model = (
                    path
                    + f"version_{i + version_offset}"
                    + "/checkpoints/"
                    + os.listdir(
                        path + f"version_{i + version_offset}" + "/checkpoints/"
                    )[0]
                )
                self.models.append(
                    DistributionalDeepNeuralNetworkNormal.load_from_checkpoint(
                        path_model
                    )
                )
        if models_checkpoints is not None:
            for i in range(n_models):
                self.models.append(
                    DistributionalDeepNeuralNetworkNormal.load_from_checkpoint(
                        models_checkpoints[i + version_offset]
                    )
                )

    @torch.no_grad()
    def _forward(self, x):
        output = [model.forward(x) for model in self.models]
        mu, logs2 = zip(*output, strict=False)
        mu = torch.stack(mu)
        logs2 = torch.stack(logs2)
        return mu, logs2

    @torch.no_grad()
    def inference(self, x):
        predictions = torch.stack([model.inference(x) for model in self.models], dim=0)
        return predictions.mean(dim=0)

    @torch.no_grad()
    def ppf(
        self,
        quantile,
        x,
        xtol=1e-5,
    ):
        output = [model.forward(x) for model in self.models]

        mu, logs2 = zip(*output, strict=False)
        mu = torch.stack(mu)
        sigma = torch.stack(logs2).exp().sqrt()

        offset = torch.stack(
            [model.hparams.output_postprocessing[0] for model in self.models]
        )[:, None, :]
        scale = torch.stack(
            [model.hparams.output_postprocessing[1] for model in self.models]
        )[:, None, :]

        mu = mu * scale + offset
        sigma = sigma * scale

        mu = mu.detach().numpy().astype(np.float64)
        sigma = sigma.detach().numpy().astype(np.float64)

        if mu.shape[0] == 1:
            return ndtri(quantile)[:, None, None] * sigma + mu

        if self.hparams.combination_approach == "gm":
            quantiles = ppf_gm(mu, sigma, quantile, xtol)
        elif self.hparams.combination_approach == "avq":
            quantiles = ppf_avq(mu, sigma, quantile)
        else:
            raise ValueError(
                f"Unknown combination approach: {self.hparams.combination_approach}"
            )

        return quantiles


def ppf_gm(mu, sigma, quantile, xtol=1e-5, coeffs=None):
    # Flatten the grid for parallel processing
    def compute_quantile_per_grid(i, j) -> np.ndarray:
        from scipy.special import ndtri

        mu_vals = mu[:, i, j]
        sigma_vals = sigma[:, i, j]
        if coeffs is not None:
            coeffs_vals = coeffs[:, i, j]
            func = lambda x, q: (
                np.sum(coeffs_vals * ndtr((x - mu_vals) / sigma_vals)) - q
            )
        else:
            func = lambda x, q: np.mean(ndtr((x - mu_vals) / sigma_vals)) - q  # noqa: E731

        result = np.zeros(len(quantile))
        for k, q in enumerate(quantile):
            if coeffs is not None:
                x0 = np.sum(coeffs_vals * (ndtri(q) * sigma_vals + mu_vals))
            else:
                x0 = np.mean(ndtri(q) * sigma_vals + mu_vals)
            result[k] = opt.fsolve(func, x0, args=(q), xtol=xtol)[0]
        return result

    # Parallel loop over the (i, j) grid
    results = Parallel(n_jobs=-1)(
        delayed(compute_quantile_per_grid)(i, j)
        for i in range(mu.shape[1])
        for j in range(mu.shape[2])
    )

    return (
        np.array(results)
        .reshape(mu.shape[1], mu.shape[2], len(quantile))
        .transpose(2, 0, 1)
    )


def ppf_avq(mu, sigma, quantile):
    mu_mean = np.mean(mu, axis=0, keepdims=True)
    sigma_mean = np.mean(sigma, axis=0, keepdims=True)
    return ndtri(quantile)[:, None, None] * sigma_mean + mu_mean
