# LEAR  # noqa: INP001

"""LEAR training and rolling-window evaluation over multiple horizons."""

import sys

sys.path.append("src/")

import pickle

import numpy as np
from sklearn import linear_model

from model_training.data_modules.utils import EPFDataModule

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
# Convert tensors to numpy arrays
x_train = x_train.numpy()
y_train = y_train.numpy()
x_val = x_val.numpy()
y_val = y_val.numpy()
x_test = x_test.numpy()
y_test = y_test.numpy()
# Get scale and offset for denormalization
offset = data_module.offset_target.numpy()
scale = data_module.scale_target.numpy()
# Get number of samples
num_train = x_train.shape[0]
num_val = x_val.shape[0]
num_test = x_test.shape[0]
num_total = num_train + num_val + num_test

# 1) LEAR for different horizons

# combine train and val data

x_train_val = np.concatenate((x_train, x_val), axis=0)
y_train_val = np.concatenate((y_train, y_val), axis=0)
num_train_val = x_train_val.shape[0]

alpha = 0.0034079864342526534  # Lasso regularization parameter
max_iter = 10000

# 1822 values observations
# forecast the last 1822 - 1461 = 361 days
# This corresponds to the validation data

# train on train data and forecast val data, rolling window, different horizons (1461, 1092, 84, 56)
pred_val_1461 = np.zeros((num_val, 24))
for i in range(num_val):
    reg_lasso_1461 = linear_model.Lasso(alpha=alpha, max_iter=max_iter)
    unten = num_train_val - num_val - 1461 + i
    oben = num_train_val - num_val + i - 1
    reg_lasso_1461.fit(X=x_train_val[unten:oben, :], y=y_train_val[unten:oben, :])
    pred_val_1461[i, :] = reg_lasso_1461.predict(X=x_train_val[[oben + 1], :])
print("1461 window: ", np.mean(np.abs((pred_val_1461 - y_val) * scale)))

pred_val_1092 = np.zeros((num_val, 24))
for i in range(num_val):
    reg_lasso_1092 = linear_model.Lasso(alpha=alpha, max_iter=max_iter)
    unten = num_train_val - num_val - 1092 + i
    oben = num_train_val - num_val + i - 1
    reg_lasso_1092.fit(X=x_train_val[unten:oben, :], y=y_train_val[unten:oben, :])
    pred_val_1092[i, :] = reg_lasso_1092.predict(X=x_train_val[[oben + 1], :])
print("1092 window: ", np.mean(np.abs((pred_val_1092 - y_val) * scale)))


pred_val_84 = np.zeros((num_val, 24))
for i in range(num_val):
    reg_lasso_84 = linear_model.Lasso(alpha=alpha, max_iter=max_iter)
    unten = num_train_val - num_val - 84 + i
    oben = num_train_val - num_val + i - 1
    reg_lasso_84.fit(X=x_train_val[unten:oben, :], y=y_train_val[unten:oben, :])
    pred_val_84[i, :] = reg_lasso_84.predict(X=x_train_val[[oben + 1], :])
print("84 window: ", np.mean(np.abs((pred_val_84 - y_val) * scale)))

pred_val_56 = np.zeros((num_val, 24))
for i in range(num_val):
    reg_lasso_56 = linear_model.Lasso(alpha=alpha, max_iter=max_iter)
    unten = num_train_val - num_val - 56 + i
    oben = num_train_val - num_val + i - 1
    reg_lasso_56.fit(X=x_train_val[unten:oben, :], y=y_train_val[unten:oben, :])
    pred_val_56[i, :] = reg_lasso_56.predict(X=x_train_val[[oben + 1], :])
print("56 window: ", np.mean(np.abs((pred_val_56 - y_val) * scale)))

pred_val_mean = (pred_val_1461 + pred_val_1092 + pred_val_84 + pred_val_56) / 4

print("MAE on val data")
print("1461 window: ", np.mean(np.abs((pred_val_1461 - y_val) * scale)))
print("1092 window: ", np.mean(np.abs((pred_val_1092 - y_val) * scale)))
print("84 window: ", np.mean(np.abs((pred_val_84 - y_val) * scale)))
print("56 window: ", np.mean(np.abs((pred_val_56 - y_val) * scale)))
print(
    "Mean: ",
    np.mean(np.abs((pred_val_mean - y_val) * scale)),
)

# combine all data for rolling window forecasting
x_train_val_test = np.concatenate((x_train, x_val, x_test), axis=0)
y_train_val_test = np.concatenate((y_train, y_val, y_test), axis=0)

# get forecasts for test data with
# different rolling window sizes (1461, 1092, 84, 56)

pred_test_1461 = np.zeros((num_test, 24))
for i in range(num_test):
    reg_lasso_1461 = linear_model.Lasso(alpha=alpha, max_iter=max_iter)
    unten = num_total - num_test - 1461 + i
    oben = num_total - num_test + i - 1
    reg_lasso_1461.fit(
        X=x_train_val_test[unten:oben, :], y=y_train_val_test[unten:oben, :]
    )
    pred_test_1461[i, :] = reg_lasso_1461.predict(X=x_train_val_test[[oben + 1], :])
print("1461 window: ", np.mean(np.abs((pred_test_1461 - y_test) * scale)))

pred_test_1092 = np.zeros((num_test, 24))
for i in range(num_test):
    reg_lasso_1092 = linear_model.Lasso(alpha=alpha, max_iter=max_iter)
    unten = num_total - num_test - 1092 + i
    oben = num_total - num_test + i - 1
    reg_lasso_1092.fit(
        X=x_train_val_test[unten:oben, :], y=y_train_val_test[unten:oben, :]
    )
    pred_test_1092[i, :] = reg_lasso_1092.predict(X=x_train_val_test[[oben + 1], :])
print("1092 window: ", np.mean(np.abs((pred_test_1092 - y_test) * scale)))

pred_test_84 = np.zeros((num_test, 24))
for i in range(num_test):
    reg_lasso_84 = linear_model.Lasso(alpha=alpha, max_iter=max_iter)
    unten = num_total - num_test - 84 + i
    oben = num_total - num_test + i - 1
    reg_lasso_84.fit(
        X=x_train_val_test[unten:oben, :], y=y_train_val_test[unten:oben, :]
    )
    pred_test_84[i, :] = reg_lasso_84.predict(X=x_train_val_test[[oben + 1], :])
print("84 window: ", np.mean(np.abs((pred_test_84 - y_test) * scale)))

pred_test_56 = np.zeros((num_test, 24))
for i in range(num_test):
    reg_lasso_56 = linear_model.Lasso(alpha=alpha, max_iter=max_iter)
    unten = num_total - num_test - 56 + i
    oben = num_total - num_test + i - 1
    reg_lasso_56.fit(
        X=x_train_val_test[unten:oben, :], y=y_train_val_test[unten:oben, :]
    )
    pred_test_56[i, :] = reg_lasso_56.predict(X=x_train_val_test[[oben + 1], :])
print("56 window: ", np.mean(np.abs((pred_test_56 - y_test) * scale)))

pred_test_mean = (pred_test_1461 + pred_test_1092 + pred_test_84 + pred_test_56) / 4

print("MAE on test data")
print("1461 window: ", np.mean(np.abs(pred_test_1461 - y_test) * scale))
print("1092 window: ", np.mean(np.abs(pred_test_1092 - y_test) * scale))
print("84 window: ", np.mean(np.abs(pred_test_84 - y_test) * scale))
print("56 window: ", np.mean(np.abs(pred_test_56 - y_test) * scale))
print(
    "Mean: ",
    np.mean(np.abs(pred_test_mean - y_test) * scale),
)

# pred without rolling window
reg_lasso_no_roll = linear_model.Lasso(alpha=alpha, max_iter=max_iter)
reg_lasso_no_roll.fit(X=x_train, y=y_train)
pred_val_no_roll = reg_lasso_no_roll.predict(X=x_val)
pred_test_no_roll = reg_lasso_no_roll.predict(X=x_test)
print(
    "MAE on val data without rolling window: ",
    np.mean(np.abs(pred_val_no_roll - y_val) * scale),
)
print(
    "MAE on test data without rolling window: ",
    np.mean(np.abs(pred_test_no_roll - y_test) * scale),
)

# save mean-predictions
LEAR_refit_4m = {}
LEAR_refit_4m["model_name"] = "LEAR"
LEAR_refit_4m["prediction_val"] = pred_val_mean * scale + offset
LEAR_refit_4m["prediction"] = (
    np.repeat(np.expand_dims(pred_test_mean, 0), 10, 0) * scale + offset
)
LEAR_refit_4m["prediction_val_no_roll"] = pred_val_no_roll * scale + offset
LEAR_refit_4m["prediction_no_roll"] = pred_test_no_roll * scale + offset
LEAR_refit_4m["quantile"] = np.zeros((10, 99, num_test, 24))
LEAR_refit_4m["prediction_val_1461"] = pred_val_1461 * scale + offset
LEAR_refit_4m["prediction_val_1092"] = pred_val_1092 * scale + offset
LEAR_refit_4m["prediction_val_84"] = pred_val_84 * scale + offset
LEAR_refit_4m["prediction_val_56"] = pred_val_56 * scale + offset
LEAR_refit_4m["prediction_1461"] = pred_test_1461 * scale + offset
LEAR_refit_4m["prediction_1092"] = pred_test_1092 * scale + offset
LEAR_refit_4m["prediction_84"] = pred_test_84 * scale + offset
LEAR_refit_4m["prediction_56"] = pred_test_56 * scale + offset
LEAR_refit_4m["offset"] = offset
LEAR_refit_4m["scale"] = scale
LEAR_refit_4m["y_val"] = y_val * scale + offset
LEAR_refit_4m["y_test"] = y_test * scale + offset

model_dict = []
model_dict.append(LEAR_refit_4m)

with open(  # noqa: PTH123
    f"notebooks/evaluate_models/results/archive/LEAR_250816_a{alpha}_{max_iter}it.pkl",
    "wb",
) as fp:
    pickle.dump(model_dict, fp)
