"""Metric utilities for evaluating probabilistic electricity price forecasts."""

import numpy as np


def compute_crps(quantiles, model_quantile, data_labels):
    """Calculate Continuous Ranked Probability Score for quantile forecasts."""
    crps = quantiles.reshape(1, -1, 1, 1) * (data_labels - model_quantile)
    crps[np.where(data_labels < model_quantile)] = (
        (1 - quantiles).reshape(1, -1, 1, 1) * (model_quantile - data_labels)
    )[np.where(data_labels < model_quantile)]
    return np.mean(crps, axis=(1, 2, 3))


def compute_pinball(quantiles, model_quantile, data_labels):
    """Calculate Continuous Ranked Probability Score for quantile forecasts not meaned."""
    pinball = quantiles.reshape(1, -1, 1, 1) * (data_labels - model_quantile)
    pinball[np.where(data_labels < model_quantile)] = (
        (1 - quantiles).reshape(1, -1, 1, 1) * (model_quantile - data_labels)
    )[np.where(data_labels < model_quantile)]
    return pinball

def compute_pinball_naive(quantiles, model_quantile, data_labels):
    """Calculate Continuous Ranked Probability Score for quantile forecasts not meaned."""
    pinball = np.zeros(model_quantile.shape)
    for m in range(model_quantile.shape[0]):
        for q in range(model_quantile.shape[1]):
            for d in range(model_quantile.shape[2]):
                for h in range(model_quantile.shape[3]):
                    if data_labels[d, h] < model_quantile[m, q, d, h]:
                        pinball[m, q, d, h] = (1 - quantiles[q]) * (model_quantile[m, q, d, h] - data_labels[d, h])
                    else:
                        pinball[m, q, d, h] = quantiles[q] * (data_labels[d, h] - model_quantile[m, q, d, h])
    return pinball


def compute_picp(quantiles, model_quantile, data_labels):
    """Calculate Prediction Interval Coverage Probability across quantile pairs."""
    return np.flip(
        np.stack(
            [
                (
                    (model_quantile[:, i, :, :] <= data_labels)
                    & (data_labels <= model_quantile[:, -i - 1, :, :])
                ).mean(axis=(1, 2))
                for i in range(len(quantiles) // 2)
            ]
        ),
        axis=0,
    ).transpose()

def compute_mpiw(quantiles, model_quantile):
    """Calculate Mean Prediction Interval Width for each quantile pair."""
    return np.flip(
        np.stack(
            [
                np.mean(
                    (model_quantile[:, -i - 1, :, :] - model_quantile[:, i, :, :]),
                    axis=(1, 2),
                )
                for i in range(len(quantiles) // 2)
            ]
        ),
        axis=0,
    ).transpose()


if __name__ == "__main__":
    pass
