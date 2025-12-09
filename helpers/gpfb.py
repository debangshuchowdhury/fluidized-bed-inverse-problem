import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    WhiteKernel,
    RBF,
    Product,
    Kernel,
)
import pandas as pd
from typing import Optional
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from helpers import processing
import os


class GaussianProcessFB:
    """
    Gaussian Process model for pressure drop and bed height predictions and uncertainty estimation.

    Attributes:
    ----------
        kernel: The kernel used for the Gaussian Process.
        scaler: A dictionary containing scalers for input and output data.
        data: A pandas DataFrame to store training data.
        model: The Gaussian Process Regressor model.
        target_columns: List of target variable column names.
        independent_columns: List of independent variable column names.
    """

    def __init__(
        self,
        model: Optional[GaussianProcessRegressor] = None,
        kernel: Optional[Kernel] = None,
        scaler: Optional[dict] = None,
    ):
        """
        Initializes the GaussianProcessFB with specified or default kernel and scaler.

        Parameters:
        ----------
            kernel: An optional kernel for the Gaussian Process. If None, a default kernel is used.
            scaler: An optional dictionary with 'input' and 'output' scalers. If None, StandardScaler is used for both.
            model: An optional GaussianProcessRegressor model. If None, a default model is created.
        """

        if kernel is not None:
            self.kernel = kernel
        else:
            # Default kernel components: RBF, WhiteKernel, and ConstantKernel
            self.kernel = RBF(length_scale_bounds=(1e-16, 1e15)) + Product(
                WhiteKernel(noise_level_bounds=(1e-12, 1e15)),
                ConstantKernel(constant_value_bounds=(1e-20, 1e20)),
            )

        if scaler is not None:
            self.scaler = scaler
        else:
            # Default to StandardScaler for both input and output
            self.scaler = {
                "input": StandardScaler(),
                "output": StandardScaler(),
            }

        if model is not None:
            self.model = model
        else:
            # Default Gaussian Process Regressor with specified kernel
            self.model = GaussianProcessRegressor(
                kernel=self.kernel,
                n_restarts_optimizer=200,
                normalize_y=True,
                alpha=1e-10,
                # random_state=30,
            )

    def fit(
        self,
        data: pd.DataFrame,
        target_columns: list,
        independent_columns: list,
    ):
        """
        Fits the Gaussian Process model to the provided data.

        Parameters:
        ----------
            data: A pandas DataFrame containing the training data.
            target_columns: List of column names for target variables.
            independent_columns: List of column names for independent variables.
        """

        self.data = data
        self.target_columns = target_columns
        self.independent_columns = independent_columns

        X = data[independent_columns].values
        y = data[target_columns].values

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Scale input and output data
        X_scaled = self.scaler["input"].fit_transform(X)
        y_scaled = self.scaler["output"].fit_transform(y)

        # Fit the GP model
        self.model.fit(X_scaled, y_scaled)

    def predict(self, X: np.ndarray):
        """
        Predicts target variables for given input data using the trained GP model.

        Parameters:
        ----------
            X: A numpy array of input data.

        Returns:
        -------
            A tuple containing predicted values and standard deviations.
        """

        # Scale input data
        X_scaled = self.scaler["input"].transform(X)

        # Predict using the GP model
        y_pred_scaled, sigma_scaled = self.model.predict(X_scaled, return_std=True)
        if y_pred_scaled.ndim == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
            sigma_scaled = sigma_scaled.reshape(-1, 1)

        # Inverse scale the predictions
        y_pred = self.scaler["output"].inverse_transform(y_pred_scaled)
        sigma = sigma_scaled * self.scaler["output"].scale_

        return y_pred, sigma

    def get_hyperparameters(self):
        """
        Retrieves the hyperparameters of the trained GP model.

        Returns:
        -------
            A dictionary of hyperparameters.
        """
        return self.model.kernel_.get_params()

    def uncertainty_quantification(
        self,
        X: np.ndarray,
        test_data: list,
        x_axis: np.ndarray,
        pred_intercepts: np.ndarray,
    ):
        """
        Quantifies uncertainty in predictions for given input data and plots results. The results are compared against the input test data.

        Parameters:
        ----------
            X: A numpy array of input data.
            test_data: File name of test data. Single realization not used to train the model.
            x_axis: A numpy array representing the independent variable for plotting.
            pred_intercepts: A numpy array of intercepts to adjust predictions for plotting.
        """

        y_pred, sigma = self.predict(X)
        print("Predictions and uncertainties computed. Now plotting...")

        name, ext = os.path.splitext(test_data)
        if ext == ".xlsx":
            test_df = processing.read_and_process_data(test_data, isexcel=True)
        else:
            test_df = processing.read_and_process_data(test_data, isexcel=False)

        initial_bed_height = float(name.split("_")[0]) / 1000
        frequency = int(name.split("_")[3])
        step_size_fl = int(name.split("_")[1])
        step_duration = int(name.split("_")[2])
        run_type = name.split("_")[4]
        metrics = processing.filter(
            processing.calculate_metrics(
                test_df, initial_bed_height, "sand", step_size_fl
            ),
            frequency,
        )[self.target_columns + ["flowrate_combi"]]

        averages = processing.recover_averaged_data_array(
            metrics,
            frequency,
            step_size_fl,
            step_duration,
            self.target_columns + ["flowrate_combi"],
            initial_bed_height,
        )

        for i, target in enumerate(self.target_columns):
            plt.figure(figsize=(10, 6))
            # plt.scatter(
            #     test_data.iloc[:, -1],
            #     test_data.iloc[:, i] + pred_intercepts[i],
            #     # linewidth=0.5,
            #     label="Experimental Data",
            # )
            plt.scatter(
                averages["flowrate_combi"],
                averages[target] + pred_intercepts[i],
                label="Averaged Experimental Data",
                c="red",
                zorder=5,
            )
            plt.plot(
                metrics["flowrate_combi"],
                metrics[target] + pred_intercepts[i],
                color="black",
                linewidth=0.5,
                label="Measured Experimental Data",
                zorder=0,
            )
            plt.plot(
                x_axis,
                y_pred[:, i] + pred_intercepts[i],
                "b-",
                label="GP Prediction",
            )
            plt.fill_between(
                x_axis,
                y_pred[:, i] + pred_intercepts[i] - 1.96 * sigma[:, i],
                y_pred[:, i] + pred_intercepts[i] + 1.96 * sigma[:, i],
                alpha=0.2,
                color="blue",
                label="95% Confidence Interval",
            )
            plt.title(
                f"Gaussian Process Prediction with Uncertainty for {self.target_columns[i]}"
            )
            plt.xlabel("Independent Variable")
            plt.ylabel(target)
            plt.legend()
            plt.show()
