"""Machine learning module for electromagnetic field prediction.

This module provides Gaussian Process regression capabilities for predicting
electromagnetic field magnitudes using training data from FEKO simulations.
"""

import csv
import os
from pathlib import Path
from typing import Optional

import gpytorch
import numpy as np
import torch

from EMFieldML.config import config, get_logger, paths, template

logger = get_logger(__name__)

# Constants
DEFAULT_INITIAL_NOISE = 1.0 / np.sqrt(2)


class FieldGPRegressor:
    """Gaussian Process regression for electromagnetic field prediction.

    This class handles the complete ML pipeline for predicting electromagnetic
    field magnitudes using GPyTorch. It includes data loading, model training,
    and prediction capabilities for ferrite shield optimization.

    In Gaussian Process Regression, we first create a prior distribution using
    a kernel function. We assume that the prediction follows this distribution.
    Then, the parameters of the kernel function are optimized to fit the training data.

    Its key characteristic is the ability to achieve accurate learning with a
    small amount of data and a short training time.

    """

    @staticmethod
    def make_x_train_data(
        input_path_train_list: Path,
        input_path_circle: Optional[str] = None,
    ) -> np.ndarray:
        """Load and prepare training input data from FEKO simulation files.

        Args:
            input_path_train_list: Path to file containing list of training data indices
            input_path_circle: Template path for circle move data files

        Returns:
            numpy array of shape (n_samples, config.dimension_x) containing training features

        """
        if input_path_circle is None:
            input_path_circle = str(paths.CIRCLE_MOVE_X_DIR / template.circle_move_x)

        with Path(input_path_train_list).open() as f:
            init_train_data_list = [int(line.strip()) for line in f]

        x_train_data = np.empty((0, config.dimension_x), float)
        for i in range(len(init_train_data_list)):
            input_path_circle_file = input_path_circle.format(
                index=init_train_data_list[i]
            )
            with Path(input_path_circle_file).open() as f:
                data = f.readlines()
                row_data = [
                    [float(row.rstrip("\n")) for row in data]
                ]  # @HonjoYuichi: Renamed from 'l' to 'row_data' for clarity
                x_train_data = np.append(x_train_data, row_data, axis=0)
        return x_train_data

    @staticmethod
    def make_y_train_data(
        input_path_y_data: str,
        postprocessing_value: float = 1.0,
    ):
        """Create Y training data with postprocessing.

        This is the power transformation.
        This transformation helps make skewed data more closely resemble a normal distribution.

        Note:
            This transformation corresponds to Figure 7 of the paper.

        """
        y_test_data = np.empty((0, 1), float)
        with Path(input_path_y_data).open() as f:
            reader = csv.reader(f)
            csv_rows = list(reader)
            for i in range(len(csv_rows)):
                y_test_data = np.append(
                    y_test_data,
                    [[float(csv_rows[i][0]) ** (1 / postprocessing_value)]],
                    axis=0,
                )
        return y_test_data

    @staticmethod
    def prepare_model(
        input_path_y_data: Path,
        x_train_data: np.ndarray,
    ) -> tuple:
        """Prepare machine learning model for training.

        The prepared model is RBF kernel-based Gaussian Process Regression.
        """
        y_train_data = FieldGPRegressor.make_x_train_data(input_path_y_data)

        train_x = torch.from_numpy(x_train_data).clone().float()
        train_y = np.array([])
        for i in range(len(y_train_data)):
            train_y = np.append(train_y, y_train_data[i, 0])
        train_y = torch.from_numpy(train_y).clone().float()
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood)

        return model, likelihood, train_x, train_y

    @staticmethod
    def calculate_deviation(
        input_path_circle: Path,
        model,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ) -> float:
        """Calculate deviation using Gaussian Process model.

        The predictive variance, which quantifies the uncertainty of the prediction,
        is calculated simultaneously with the prediction itself in Gaussian Process Regression.
        This is done according to Equation (9) of the paper.
        """
        x_test_data = np.empty((0, config.dimension_x), float)
        with Path(input_path_circle).open() as f:
            data = f.readlines()
            test_row_data = [[float(row.rstrip("\n")) for row in data]]
            x_test_data = np.append(x_test_data, test_row_data, axis=0)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_x = torch.from_numpy(x_test_data).clone().float()
            observed_pred = likelihood(model(test_x))

        return (observed_pred.variance.numpy()[0]) ** (1 / 2)

    @staticmethod
    def learning_magneticfield(
        input_path_y_data: Path,
        output_path: Path,
        x_train_data: np.ndarray,
        lr_record: float = 0.02,
        iter_record: int = 1500,
        inter_record: int = 600,
        small_record: float = 5.0,
        initial_noise: float = DEFAULT_INITIAL_NOISE,
    ):
        """Train magnetic field learning model.

        The training algorithm is based on maximizing the likelihood method.

        Args:
            input_path_y_data: Path to training target data
            output_path: Path to save the trained model
            x_train_data: Training input data
            lr_record: Learning rate for optimizer
            iter_record: Number of training iterations
            inter_record: Interval for learning rate adjustment
            small_record: Factor for learning rate reduction
            initial_noise: Initial noise level for Gaussian likelihood

        """
        print(
            f"lr = {lr_record}, iteration = {iter_record}, interval = {inter_record}, small = {small_record}"
        )

        model, likelihood, train_x, train_y = FieldGPRegressor.prepare_model(
            input_path_y_data, x_train_data
        )

        model.mean_module.constant.data.fill_(train_y.mean())
        model.mean_module.constant.requires_grad_(False)
        likelihood.noise = initial_noise

        smoke_test = "CI" in os.environ
        training_iter = 2 if smoke_test else iter_record

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(
            [
                {
                    "params": model.parameters()
                },  # Includes GaussianLikelihood parameters
            ],
            lr=lr_record,
        )

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        losses = []
        lengthscales = []
        noises = []
        for i in range(training_iter):
            if i != 0 and i % inter_record == 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] /= small_record
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            if i == 0 or i % 100 == 99:
                print(
                    f"Iter {i + 1}/{training_iter} - Loss: {loss.item():.6f}   lengthscale: {model.covar_module.base_kernel.lengthscale.mean().item():.6f}   noise: {model.likelihood.noise.item():.6f}"
                )
            loss.backward()
            losses.append(loss.item())
            lengthscales.append(
                model.covar_module.base_kernel.lengthscale.mean().item()
            )
            noises.append(model.likelihood.noise.item())
            optimizer.step()

        # 学習後、モデルの状態を辞書として保存します。
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "likelihood_state_dict": likelihood.state_dict(),
                "optimizeer_state_dict": optimizer.state_dict(),
                "losses": losses,
                "lengthscales": lengthscales,
                "noises": noises,
            },
            output_path,
        )

        logger.info(f"Saved {output_path}")


class ExactGPModel(gpytorch.models.ExactGP):
    """Exact Gaussian Process model for magnetic field prediction."""

    def __init__(self, train_x, train_y, likelihood):
        """Initialize the ExactGP model.

        The model is based on an RBF kernel.
        In addition, the model includes scale and noise corrections.
        The details of this are based on Equation (15) in the paper.

        """
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=config.dimension_x)
        )
