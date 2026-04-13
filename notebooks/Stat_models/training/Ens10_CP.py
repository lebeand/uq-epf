# DDNN-CP with rolling window  # noqa: INP001

"""Conformal prediction post-processing for ensemble DDNN forecasts."""

import sys

sys.path.append("src/")

import pickle
from lightning import seed_everything


import numpy as np

from model_training.data_modules.utils import EPFDataModule

seed_everything(0)

# load data
data_module = EPFDataModule(
    data_file_path="data/processed/smard_data_201810010000_202501010000.npz",
    val_date="2022-12-01",
    test_date="2023-12-01",
    end_date="2024-11-30",
    batch_size=32,
    standardization_case="mean_std",
)
x_train, y_train = data_module.train_dataset[:]
x_val, y_val = data_module.val_dataset[:]
x_test, y_test = data_module.test_dataset[:]
# Get scale and offset for denormalization
offset = data_module.offset_target.numpy()
scale = data_module.scale_target.numpy()

# load model
model_name = "DDNN_Ens"
with open(f"notebooks/evaluate_models/results/metric_evaluation/{model_name}.pkl", "rb") as f:  # noqa: PTH123
    model_dict = pickle.load(f)  # noqa: S301
model_name = "Ens_CP"

Ens10 = model_dict[2]

pred_val_list = []
pred_test_list = []
for model in Ens10["model"]:
    pred_val_list.append((np.array(model.inference(x_val))-offset) / scale)
    pred_test_list.append((np.array(model.inference(x_test))-offset) / scale)

y_val = y_val.numpy()
y_test = y_test.numpy()
num_val = y_val.shape[0]
num_test = y_test.shape[0]

percentiles_test_list = []
for pred_val, pred_test in zip(pred_val_list, pred_test_list, strict=True):

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

    percentiles_test_list.append(percentiles_test)


# compute crps over percentiles (from Andreas)
def compute_crps(quantiles, model_quantile, data_labels):
    crps = quantiles.reshape(1, -1, 1, 1) * (data_labels - model_quantile)

    crps[np.where(data_labels < model_quantile)] = (
        (1 - quantiles).reshape(1, -1, 1, 1) * (model_quantile - data_labels)
    )[np.where(data_labels < model_quantile)]

    return np.mean(crps, axis=(1, 2, 3))



quantiles = np.array([i / 100 for i in range(1, 100)])
pred_test = np.stack(pred_test_list, axis=0) * scale + offset
percentiles_test = np.stack(percentiles_test_list, axis=0) * scale + offset

# with updated coef
print(
    compute_crps(
        quantiles=quantiles,
        model_quantile=percentiles_test,
        data_labels=y_test * scale + offset,
    )
)

# save percentiles and mean-predictions
DDNN_CP = {}
DDNN_CP["model_name"] = "Ens10-CP"
DDNN_CP["prediction"] = pred_test
DDNN_CP["quantile"] = percentiles_test

model_dict = []
model_dict.append(DDNN_CP)

with open(  # noqa: PTH123
    f"notebooks/evaluate_models/results/archive/{model_name}.pkl",
    "wb",
) as fp:
    pickle.dump(model_dict, fp)
