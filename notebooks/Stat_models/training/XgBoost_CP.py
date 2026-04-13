# XGBoost-CP with rolling window  # noqa: INP001

"""Conformal prediction interval construction for XGBoost forecasts."""

import sys

sys.path.append("src/")

import pickle

import numpy as np

# load model
model_name = "XGBoost"
with open(f"notebooks/evaluate_models/results/archive/{model_name}.pkl", "rb") as f:  # noqa: PTH123
    model_dict = pickle.load(f)  # noqa: S301
model_name = model_name.replace("XGBoost", "XGBoost_CP")

XGBoost_model = model_dict[0]
offset = XGBoost_model["offset"]
scale = XGBoost_model["scale"]
pred_val = (XGBoost_model["prediction_val"] - offset) / scale
pred_test = (XGBoost_model["prediction"][0] - offset) / scale

y_val = (XGBoost_model["y_val"] - offset) / scale
y_test = (XGBoost_model["y_test"] - offset) / scale
num_val = y_val.shape[0]
num_test = y_test.shape[0]

# get residuals
residuals_val = pred_val - y_val
residuals_test = pred_test - y_test

# Conformal prediction approach
# Use residuals to compute nonconformity scores (absolute residuals)
nonconformity_scores = np.concatenate([np.abs(residuals_val), np.abs(residuals_test)])

# calculate percentiles using conformal prediction
quantiles = np.array([i / 100 for i in range(1, 100)])
percentiles_test = np.zeros((len(quantiles), num_test, 24))

# compute percentiles
for i in range(num_test):
    for h in range(24):
        # Get the most recent residuals for this hour
        scores = nonconformity_scores[i : num_val + i, h]

        # Compute empirical quantiles from nonconformity scores
        for q_idx, q in enumerate(quantiles):
            # Following the conformal prediction formula: q_hat(alpha|p_hat_d,h)
            if q < 0.5:
                score_quantile = np.quantile(scores, 1 - 2 * q)
                percentiles_test[q_idx, i, h] = pred_test[i, h] - score_quantile
            else:
                score_quantile = np.quantile(scores, 2 * q - 1)
                percentiles_test[q_idx, i, h] = pred_test[i, h] + score_quantile


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
XGBoost_CP = {}
XGBoost_CP["model_name"] = "XGBoost-CP"
XGBoost_CP["prediction"] = pred_test
XGBoost_CP["quantile"] = percentiles_test

model_dict = []
model_dict.append(XGBoost_CP)

with open(  # noqa: PTH123
    f"notebooks/evaluate_models/results/archive/{model_name}.pkl",
    "wb",
) as fp:
    pickle.dump(model_dict, fp)
