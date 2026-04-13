# LEAR-QRA with rolling window  # noqa: INP001

"""Quantile regression averaging calibration for LEAR forecast ensembles."""

import sys

sys.path.append("src/")

import pickle

import numpy as np
from sklearn.linear_model import QuantileRegressor

# Quantile regression

# load model
model_name = "LEAR"
with open(f"notebooks/evaluate_models/results/archive/{model_name}.pkl", "rb") as f:  # noqa: PTH123
    model_dict = pickle.load(f)  # noqa: S301
model_name = model_name.replace("LEAR", "LEAR_QRA")

LEAR = model_dict[0]
offset = LEAR["offset"]
scale = LEAR["scale"]
pred_val_1461 = (LEAR["prediction_val_1461"] - offset) / scale
pred_val_1092 = (LEAR["prediction_val_1092"] - offset) / scale
pred_val_84 = (LEAR["prediction_val_84"] - offset) / scale
pred_val_56 = (LEAR["prediction_val_56"] - offset) / scale
pred_test_1461 = (LEAR["prediction_1461"] - offset) / scale
pred_test_1092 = (LEAR["prediction_1092"] - offset) / scale
pred_test_84 = (LEAR["prediction_84"] - offset) / scale
pred_test_56 = (LEAR["prediction_56"] - offset) / scale

y_val = (LEAR["y_val"] - offset) / scale
y_test = (LEAR["y_test"] - offset) / scale
num_test = y_test.shape[0]

quant_X = np.array((pred_val_1461, pred_val_1092, pred_val_84, pred_val_56))
quant_X = np.transpose(quant_X, axes=(1, 0, 2))
quantiles = np.array([(i + 1) / 100 for i in range(99)])

# iterate over series and quantiles:
coefs = np.zeros((5, 99, 24))  # 5 parameters, 99 quantiles, 24 series
for i in range(24):
    for j in range(99):
        Qreg = QuantileRegressor(
            quantile=quantiles[j].item(), fit_intercept=True, alpha=0.0
        )
        Qreg.fit(X=quant_X[:, :, i], y=y_val[:, i])
        coefs[0, j, i] = Qreg.intercept_
        coefs[1:5, j, i] = Qreg.coef_

# # numerical instability at 4th time series
# # Fitting without intercept resolves the problem
# Qreg = QuantileRegressor(quantile=quantiles[55].item(), fit_intercept=False)
# Qreg.fit(X = quant_X[:,:,4], y = y_val[:,4])
# # overwrite the coefficients
# coefs[0,55,4] = Qreg.intercept_
# coefs[1:5,55,4] = Qreg.coef_

# combine forecasts in array
pred_test = np.array(
    (
        np.ones((num_test, 24)),
        pred_test_1461,
        pred_test_1092,
        pred_test_84,
        pred_test_56,
    )
)
pred_test_t = np.transpose(pred_test, axes=(1, 2, 0))
coefs_t = np.transpose(coefs, axes=(0, 2, 1))

# calculate percentiles
percentiles_test = np.zeros((99, num_test, 24))
for h in range(24):
    for d in range(num_test):
        for q in range(99):
            percentiles_test[q, d, h] = np.matmul(
                coefs_t[:, h, q], pred_test_t[d, h, :]
            )


# compute crps over percentiles (from Andreas)
def compute_crps(quantiles, model_quantile, data_labels):
    crps = quantiles.reshape(1, -1, 1, 1) * (data_labels - model_quantile)

    crps[np.where(data_labels < model_quantile)] = (
        (1 - quantiles).reshape(1, -1, 1, 1) * (model_quantile - data_labels)
    )[np.where(data_labels < model_quantile)]

    return np.mean(crps, axis=(1, 2, 3))


quantiles = np.array([i / 100 for i in range(1, 100)])

pred_test_mean = (
    np.repeat(np.expand_dims(np.mean(pred_test[1:5, :, :], axis=0), 0), 10, 0) * scale
    + offset
)
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
# 5.38379206


# save percentiles and mean-predictions
LEAR_QRA = {}
LEAR_QRA["model_name"] = "LEAR_QRA"
LEAR_QRA["prediction"] = pred_test_mean
LEAR_QRA["quantile"] = percentiles_test

model_dict = []
model_dict.append(LEAR_QRA)

with open(  # noqa: PTH123s
    f"notebooks/evaluate_models/results/archive/{model_name}.pkl",
    "wb",
) as fp:
    pickle.dump(model_dict, fp)
