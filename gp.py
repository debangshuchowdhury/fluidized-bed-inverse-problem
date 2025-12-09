from helpers.gpfb import GaussianProcessFB as GPFB
import pandas as pd
import numpy as np


num_pred = 700
L2 = 0
targets = ["bed_exp"]  # ["p1"]  # ,
intercepts = np.array([0.956])  # , 0.808])
independent_var = ["fl_L1", "fl_L2"]  # , "initial_bed_height"]
training_data = pd.read_csv("sand_data.csv")

# Constructing input space for predictions
flowrate_range = np.vstack(
    [
        np.linspace(0, 600, num_pred).ravel(),
        np.linspace(0, 600, num_pred).ravel() * L2,
    ]
).T
additional = np.vstack(
    [np.linspace(600, 1600, 300).ravel(), np.ones(300).ravel() * 600 * L2]
).T
flowrate_range = np.vstack([flowrate_range, additional])
totalflowrate = flowrate_range.sum(axis=1)
prediction_inputs = np.hstack(
    [
        flowrate_range,
        # np.ones(num_pred + 300).reshape(num_pred + 300, 1) * 0.808,
    ]
)

# Training the model
modelc = GPFB()
modelc.fit(training_data, targets, independent_var)

print("Model trained. Now predicting...")

exp = "956_40_120_50_forward.xlsx"  #
# exp = "808_10_30_15_hysteresis.csv"  #

# Uncertainty quantification and plotting
modelc.uncertainty_quantification(
    X=prediction_inputs,
    test_data=exp,
    x_axis=totalflowrate,
    pred_intercepts=intercepts,
)

print("done")
