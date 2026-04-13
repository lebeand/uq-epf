# LEAR-GARCH with rolling window  # noqa: INP001

"""GARCH-based residual volatility calibration for LEAR forecasts."""

import sys

sys.path.append("src/")

import pickle

import numpy as np
from arch import arch_model
from scipy.stats import norm

# load model
model_name = "LEAR"
with open(f"notebooks/evaluate_models/results/archive/{model_name}.pkl", "rb") as f:  # noqa: PTH123
    model_dict = pickle.load(f)  # noqa: S301
model_name = model_name.replace("LEAR", "LEAR_GARCH")

LEAR = model_dict[0]
offset = LEAR["offset"]
scale = LEAR["scale"]
pred_val = (LEAR["prediction_val"] - offset) / scale
pred_test = (LEAR["prediction"][0] - offset) / scale

y_val = (LEAR["y_val"] - offset) / scale
y_test = (LEAR["y_test"] - offset) / scale
num_val = y_val.shape[0]
num_test = y_test.shape[0]

# get residuals
residuals_val = pred_val - y_val
residuals_test = pred_test - y_test
residuals_val_test = np.concatenate((residuals_val, residuals_test), axis=0)

# GARCH on residuals
sigma_garch = np.zeros((num_test, 24))

# iterate over 24 time series
for i in range(num_test):
    for h in range(24):
        garch_model = arch_model(
            residuals_val_test[i : num_val + i, h],
            mean="Zero",
            vol="Garch",
            p=1,
            q=1,
            dist="normal",
            rescale=False,
        )
        garch_fit = garch_model.fit(disp="off")
        f0 = garch_fit.forecast()
        sigma_garch[i, h] = np.sqrt(np.array(f0.variance.values[-1, 0]))

# calculate percentiles using normal distribution with GARCH volatility
quantiles = np.array([i / 100 for i in range(1, 100)])
percentiles_test = np.zeros((len(quantiles), num_test, 24))

for i in range(num_test):
    for h in range(24):
        for q_idx, q in enumerate(quantiles):
            percentiles_test[q_idx, i, h] = (
                pred_test[i, h] + norm.ppf(q) * sigma_garch[i, h]
            )


# compute crps over percentiles (from Andreas)
def compute_crps(quantiles, model_quantile, data_labels):
    crps = quantiles.reshape(1, -1, 1, 1) * (data_labels - model_quantile)

    crps[np.where(data_labels < model_quantile)] = (
        (1 - quantiles).reshape(1, -1, 1, 1) * (model_quantile - data_labels)
    )[np.where(data_labels < model_quantile)]

    return np.mean(crps, axis=(1, 2, 3))


quantiles = np.array([i / 100 for i in range(1, 100)])

pred_test = np.repeat(np.expand_dims(pred_test, 0), 10, 0) * scale + offset
percentiles_test = (
    np.repeat(np.expand_dims(percentiles_test, axis=0), 10, 0) * scale + offset
)

# with updated coef
print(
    compute_crps(
        quantiles=quantiles,
        model_quantile=percentiles_test,
        data_labels=y_test * scale + offset,
    )
)

# save percentiles and mean-predictions
LEAR_GARCH = {}
LEAR_GARCH["model_name"] = "LEAR_GARCH"
LEAR_GARCH["prediction"] = pred_test
LEAR_GARCH["quantile"] = percentiles_test

model_dict = []
model_dict.append(LEAR_GARCH)

with open(  # noqa: PTH123
    f"notebooks/evaluate_models/results/archive/{model_name}.pkl",
    "wb",
) as fp:
    pickle.dump(model_dict, fp)
