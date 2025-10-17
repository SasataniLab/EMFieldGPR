"""Gaussian Process prediction utilities for electromagnetic field analysis.

This module provides functions for Gaussian Process regression and prediction
in electromagnetic field modeling and visualization.


"""

import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from tqdm import tqdm

from EMFieldML.config import config, get_logger, paths, template
from EMFieldML.Learning.Learning import FieldGPRegressor

logger = get_logger(__name__)


# RBFカーネル関数
def rbf_kernel(X1, X2, lengthscale, scale):
    """Calculate RBF kernel matrix for Gaussian Process regression."""
    pairwise_dists = cdist(X1 / lengthscale, X2 / lengthscale, "sqeuclidean")
    return scale * np.exp(-0.5 * pairwise_dists)


# 平均関数を定義
def mean_function(mean_constant, X):
    """Calculate mean function for Gaussian Process regression."""
    return mean_constant * np.ones(X.shape[0])


def mean_function_one(mean_constant, X):
    """Calculate mean function for single point prediction."""
    return mean_constant * np.ones(X)


# mu_starを計算する関数
def calculate_mu_star(X_test, k_star, K_inv_Y_train, mean_constant):
    """Calculate mu_star for Gaussian Process prediction."""
    ans = []
    for i in range(len(K_inv_Y_train)):
        mu_star = mean_function(mean_constant, X_test) + k_star.T @ K_inv_Y_train[i]
        ans.append(mu_star)
    return np.array(ans)


def calculate_mu_star_one(k_star, K_inv_Y_train, mean_constant):
    """Calculate mu_star for single point prediction."""
    mu_star = mean_function_one(mean_constant, 1) + k_star.T @ K_inv_Y_train
    return np.array(mu_star)


def make_y_data(Y_train, input_path, preprocessing_value):
    """Load and preprocess Y training data from CSV file.

    Args:
        Y_train: Training data array (unused parameter -
        input_path: Path to CSV file containing target values
        preprocessing_value: Value used for power transformation (1/preprocessing_value)

    Returns:
        numpy.ndarray: Preprocessed training data with shape (n_samples, 1)

    Note:
        Please verify if this is the correct preprocessing approach for the target values.

    """
    y_train_data = np.empty((0, 1), float)
    with Path(input_path).open() as f:
        reader = csv.reader(f)
        data_rows = list(reader)  #
        for j in range(len(data_rows)):
            y_train_data = np.append(
                y_train_data,
                [[float(data_rows[j][0]) ** (1 / preprocessing_value)]],
                axis=0,
            )
    Y_train_mag_one = np.array([])
    for j in range(config.n_train_data):
        Y_train_mag_one = np.append(Y_train_mag_one, y_train_data[j, 0])
    Y_train.append(Y_train_mag_one)

    return Y_train


def calculate_K_inv_Y_train(Y_train, K_inv, mean_constant, x_train_data):
    """Calculate K_inv_Y_train for Gaussian Process regression."""
    K_inv_Y_train = []
    for i in range(len(Y_train)):
        k_inv_y_train = K_inv @ (
            Y_train[i] - mean_function(mean_constant, x_train_data)
        )
        K_inv_Y_train.append(k_inv_y_train)
    return np.array(K_inv_Y_train)


# 予測の前にできる処理をする関数
def prediction_prepare(
    prediction_point_number: int = config.n_prediction_points_level2,
):
    """Prepare data for Gaussian Process prediction."""
    # 用意するテストデータ
    Y_train_mag = []
    Y_train_phi = []
    Y_train_theta = []

    x_train_data = FieldGPRegressor.make_x_train_data(
        paths.TRAIN_DIR / template.x_train_after_active_learning
    )

    for i in tqdm(
        range(prediction_point_number), desc="Loading magnitude files", unit="files"
    ):
        input_path = paths.Y_DATA_DIR / template.y_data_magnitude.format(index=i)
        Y_train_mag = make_y_data(
            Y_train_mag, input_path, config.prepocessing_value_magnitude
        )

    Y_train_mag = np.array(Y_train_mag)

    for i in tqdm(
        range(prediction_point_number), desc="Loading theta files", unit="files"
    ):
        input_path = paths.Y_DATA_DIR / template.y_data_vector_theta.format(index=i)
        Y_train_theta = make_y_data(
            Y_train_theta, input_path, config.prepocessing_value_vector
        )

    Y_train_theta = np.array(Y_train_theta)

    for i in tqdm(
        range(prediction_point_number), desc="Loading phi files", unit="files"
    ):
        input_path = paths.Y_DATA_DIR / template.y_data_vector_phi.format(index=i)
        Y_train_phi = make_y_data(
            Y_train_phi, input_path, config.prepocessing_value_vector
        )

    Y_train_phi = np.array(Y_train_phi)

    # -------- モデルから関数の値を抽出し、利用する -------- #
    checkpoint = torch.load(paths.LEARNING_MODEL_DIR / template.model)
    model_state_dict = checkpoint["model_state_dict"]
    lengthscale = np.log(
        1 + np.exp(model_state_dict["covar_module.base_kernel.raw_lengthscale"])
    )
    raw_noise = model_state_dict["likelihood.noise_covar.raw_noise"].numpy()
    noise = np.log(1 + np.exp(raw_noise).item())
    scale = np.log(1 + np.exp(model_state_dict["covar_module.raw_outputscale"].numpy()))
    mean_constant_mag = Y_train_mag.mean()
    mean_constant_theta = Y_train_theta.mean()
    mean_constant_phi = Y_train_phi.mean()
    K = rbf_kernel(x_train_data, x_train_data, lengthscale, scale) + noise * np.eye(
        len(x_train_data)
    )
    K_inv = cho_solve(cho_factor(K), np.eye(len(x_train_data)))

    K_inv_Y_train_mag = calculate_K_inv_Y_train(
        Y_train_mag, K_inv, mean_constant_mag, x_train_data
    )

    K_inv_Y_train_theta = calculate_K_inv_Y_train(
        Y_train_theta, K_inv, mean_constant_theta, x_train_data
    )

    K_inv_Y_train_phi = calculate_K_inv_Y_train(
        Y_train_phi, K_inv, mean_constant_phi, x_train_data
    )

    logger.info("✅ ML models loaded successfully")
    return (
        x_train_data,
        K_inv_Y_train_mag,
        K_inv_Y_train_theta,
        K_inv_Y_train_phi,
        lengthscale,
        scale,
        K_inv,
        mean_constant_mag,
        mean_constant_theta,
        mean_constant_phi,
    )


def prediction_prepare_efficiency():
    """Prepare data for efficiency prediction."""
    # 用意するテストデータ
    Y_train_efficiency = []

    x_train_data = FieldGPRegressor.make_x_train_data(
        paths.TRAIN_DIR / template.x_train_after_active_learning
    )

    input_path = paths.Y_DATA_DIR / template.y_data_efficiency
    Y_train_efficiency = make_y_data(
        Y_train_efficiency, input_path, config.prepocessing_value_efficiency
    )
    Y_train_efficiency = np.array(Y_train_efficiency)

    # -------- モデルから関数の値を抽出し、利用する -------- #
    checkpoint = torch.load(paths.LEARNING_MODEL_DIR / template.model)
    model_state_dict = checkpoint["model_state_dict"]
    lengthscale = np.log(
        1 + np.exp(model_state_dict["covar_module.base_kernel.raw_lengthscale"])
    )
    raw_noise = model_state_dict["likelihood.noise_covar.raw_noise"].numpy()
    noise = np.log(1 + np.exp(raw_noise).item())
    scale = np.log(1 + np.exp(model_state_dict["covar_module.raw_outputscale"].numpy()))
    mean_constant = Y_train_efficiency.mean()
    K = rbf_kernel(x_train_data, x_train_data, lengthscale, scale) + noise * np.eye(
        len(x_train_data)
    )
    K_inv = cho_solve(cho_factor(K), np.eye(len(x_train_data)))

    K_inv_Y_train_efficiency = calculate_K_inv_Y_train(
        Y_train_efficiency, K_inv, mean_constant, x_train_data
    )

    logger.info("✅ Efficiency models loaded successfully")
    return (
        x_train_data,
        K_inv_Y_train_efficiency,
        lengthscale,
        scale,
        K_inv,
        mean_constant,
        noise,
    )


# mu_starを計算するのに特化した関数
def mu_star(X_train, X_test, K_inv_Y_train, lengthscale, scale, mean_constant):
    """Calculate mu_star for Gaussian Process prediction."""
    k_star = rbf_kernel(X_train, X_test, lengthscale, scale)
    return calculate_mu_star(X_test, k_star, K_inv_Y_train, mean_constant) ** (
        config.prepocessing_value_magnitude
    )


def mu_star_one(
    X_train,
    X_test,
    K_inv_Y_train,
    lengthscale,
    scale,
    mean_constant,
    preprocessing_value,
):
    """Calculate mu_star for single point prediction."""
    k_star = rbf_kernel(X_train, X_test, lengthscale, scale)
    return calculate_mu_star_one(k_star, K_inv_Y_train, mean_constant) ** (
        preprocessing_value
    )


def mu_star_efficiency(
    X_train, X_test, K_inv_Y_train, lengthscale, scale, K_inv, mean_constant, noise
):
    """Calculate mu_star for efficiency prediction."""
    k_star = rbf_kernel(X_train, X_test, lengthscale, scale)
    k_star_star = rbf_kernel(X_test, X_test, lengthscale, scale) + noise * np.eye(
        len(X_test)
    )
    mu_star = calculate_mu_star(X_test, k_star, K_inv_Y_train, mean_constant) ** (
        config.prepocessing_value_efficiency
    )
    var_star = (k_star_star - k_star.T @ K_inv @ k_star) ** (1 / 2)
    return mu_star, var_star


def vector(
    X_train,
    X_test,
    K_inv_Y_train_theta,
    K_inv_Y_train_phi,
    lengthscale,
    scale,
    mean_constant_theta,
    mean_constant_phi,
):
    """Calculate vector field for electromagnetic field prediction."""
    k_star = rbf_kernel(X_train, X_test, lengthscale, scale)
    theta = calculate_mu_star(X_test, k_star, K_inv_Y_train_theta, mean_constant_theta)
    phi = calculate_mu_star(X_test, k_star, K_inv_Y_train_phi, mean_constant_phi)
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)
    # The current implementation uses z = sin(phi) which may be incorrect.
    # Standard conversion should be z = cos(theta).
    return np.array([x, y, z]).T


def vector_one(phi, theta):
    """Convert spherical coordinates to Cartesian coordinates for single values.

    Args:
        phi: Azimuthal angle in radians
        theta: Polar angle in radians

    Returns:
        numpy.ndarray: Cartesian coordinates [x, y, z]

    """
    x = np.cos(phi) * np.cos(theta)
    y = np.cos(phi) * np.sin(theta)
    z = np.sin(phi)
    return np.array([x, y, z])


def calculate_error_no_devide(a, b):
    """Calculate simple difference between two values.

    Args:
        a: First value
        b: Second value

    Returns:
        float: Difference ``(a - b)``

    """
    return a - b


def calculate_error(a, b):
    """Calculate relative error between two values.

    Args:
        a: Predicted value
        b: True value

    Returns:
        float: Relative error ``|a - b| / |b|``

    """
    return abs(a - b) / abs(b)


def calculate_error_square(a, b):
    """Calculate squared error between two values.

    Args:
        a: Predicted value
        b: True value

    Returns:
        float: Squared error ``(a - b)²``

    """
    return (a - b) ** 2


def pre_magneticfield(
    model_path: Path,
    input_path_train_list: Path,
    input_path_test_list: Path,
    input_path_y_train_data: str,
    input_path_y_test_data: str,
    output_path: Path,
    func_calculate_error: Callable[[float, float], float],
    n_point: int = None,
    input_path_circle: str = None,
    input_path_circle_test: str = None,
    preprocessing_value: float = 1.0,
    noise_scale_change: bool = False,
):
    """Predict magnetic field using Gaussian Process regression."""
    # Set defaults if not provided
    if n_point is None:
        n_point = config.n_prediction_points_level1
    if input_path_circle is None:
        input_path_circle = str(paths.CIRCLE_MOVE_X_DIR / template.circle_move_x)
    if input_path_circle_test is None:
        input_path_circle_test = str(paths.CIRCLE_MOVE_X_DIR / template.circle_move_x)

    checkpoint = torch.load(model_path)

    model_state_dict = checkpoint["model_state_dict"]
    lengthscale = np.log(
        1 + np.exp(model_state_dict["covar_module.base_kernel.raw_lengthscale"])
    )
    model_state_dict = checkpoint["model_state_dict"]
    lengthscale = np.log(
        1 + np.exp(model_state_dict["covar_module.base_kernel.raw_lengthscale"])
    )
    noise = 1e-4
    scale = np.log(2)
    if noise_scale_change:
        raw_noise = model_state_dict["likelihood.noise_covar.raw_noise"].numpy()
        noise = np.log(1 + np.exp(raw_noise).item())
        scale = np.log(
            1 + np.exp(model_state_dict["covar_module.raw_outputscale"].numpy())
        )

    x_train_data = FieldGPRegressor.make_x_train_data(
        input_path_train_list=input_path_train_list,
        input_path_circle=input_path_circle,
    )
    x_test_data = FieldGPRegressor.make_x_train_data(
        input_path_train_list=input_path_test_list,
        input_path_circle=input_path_circle_test,
    )
    n_test = len(x_test_data)

    K = rbf_kernel(x_train_data, x_train_data, lengthscale, scale) + noise * np.eye(
        len(x_train_data)
    )
    K_inv = cho_solve(cho_factor(K), np.eye(len(x_train_data)))

    error_distribution = [0 for _ in range(n_test)]
    for i in range(n_point):
        y_train_data = FieldGPRegressor.make_y_train_data(
            input_path_y_train_data.format(index=i), preprocessing_value
        )
        y_test_data = FieldGPRegressor.make_y_train_data(
            input_path_y_test_data.format(index=i)
        )

        mean_constant = y_train_data.mean()

        K_inv_Y_train_mag = K_inv @ (y_train_data - mean_function_one(mean_constant, 1))

        for k in range(n_test):
            predicted_magnitude = mu_star_one(
                x_train_data,
                x_test_data[k],
                K_inv_Y_train_mag,
                lengthscale,
                scale,
                mean_constant,
                preprocessing_value,
            )
            error = func_calculate_error(predicted_magnitude[0][0], y_test_data[k][0])
            error_distribution[k] += float(error)
    error_distribution = np.array(error_distribution)
    error_distribution /= float(n_point)

    with Path(output_path).open("w") as f:
        for i in range(len(error_distribution)):
            print(error_distribution[i], file=f)

    logger.info(f"Saved {output_path}")


def pre_vector(
    model_path: Path,
    input_path_train_list: Path,
    input_path_test_list: Path,
    input_path_y_train_data_phi: str,
    input_path_y_train_data_theta: str,
    input_path_y_test_data_phi: str,
    input_path_y_test_data_theta: str,
    output_path: Path,
    n_point: int = None,
    input_path_circle: str = None,
    input_path_circle_test: str = None,
    preprocessing_value: float = 1.0,
    noise_scale_change: bool = False,
):
    """Predict vector field using Gaussian Process regression."""
    # Set defaults if not provided
    if n_point is None:
        n_point = config.n_prediction_points_level1
    if input_path_circle is None:
        input_path_circle = str(paths.CIRCLE_MOVE_X_DIR / template.circle_move_x)
    if input_path_circle_test is None:
        input_path_circle_test = str(paths.CIRCLE_MOVE_X_DIR / template.circle_move_x)

    checkpoint = torch.load(model_path)

    model_state_dict = checkpoint["model_state_dict"]
    lengthscale = np.log(
        1 + np.exp(model_state_dict["covar_module.base_kernel.raw_lengthscale"])
    )
    model_state_dict = checkpoint["model_state_dict"]
    lengthscale = np.log(
        1 + np.exp(model_state_dict["covar_module.base_kernel.raw_lengthscale"])
    )
    noise = 1e-4
    scale = np.log(2)
    if noise_scale_change:
        raw_noise = model_state_dict["likelihood.noise_covar.raw_noise"].numpy()
        noise = np.log(1 + np.exp(raw_noise).item())
        scale = np.log(
            1 + np.exp(model_state_dict["covar_module.raw_outputscale"].numpy())
        )

    x_train_data = FieldGPRegressor.make_x_train_data(
        input_path_train_list=input_path_train_list,
        input_path_circle=input_path_circle,
    )
    x_test_data = FieldGPRegressor.make_x_train_data(
        input_path_train_list=input_path_test_list,
        input_path_circle=input_path_circle_test,
    )
    n_test = len(x_test_data)

    K = rbf_kernel(x_train_data, x_train_data, lengthscale, scale) + noise * np.eye(
        len(x_train_data)
    )
    K_inv = cho_solve(cho_factor(K), np.eye(len(x_train_data)))

    error_distribution = [0 for _ in range(n_test)]
    for i in range(n_point):
        y_train_data_phi = FieldGPRegressor.make_y_train_data(
            input_path_y_train_data_phi.format(index=i), preprocessing_value
        )
        y_train_data_theta = FieldGPRegressor.make_y_train_data(
            input_path_y_train_data_theta.format(index=i), preprocessing_value
        )
        y_test_data_phi = FieldGPRegressor.make_y_train_data(
            input_path_y_test_data_phi.format(index=i), preprocessing_value
        )
        y_test_data_theta = FieldGPRegressor.make_y_train_data(
            input_path_y_test_data_theta.format(index=i), preprocessing_value
        )

        mean_constant_phi = y_train_data_phi.mean()
        mean_constant_theta = y_train_data_theta.mean()

        K_inv_Y_train_phi = K_inv @ (
            y_train_data_phi - mean_function_one(mean_constant_phi, 1)
        )
        K_inv_Y_train_theta = K_inv @ (
            y_train_data_theta - mean_function_one(mean_constant_theta, 1)
        )

        for k in range(n_test):
            predicted_vector_phi = mu_star_one(
                x_train_data,
                x_test_data[k],
                K_inv_Y_train_phi,
                lengthscale,
                scale,
                mean_constant_phi,
                preprocessing_value,
            )
            predicted_vector_theta = mu_star_one(
                x_train_data,
                x_test_data[k],
                K_inv_Y_train_theta,
                lengthscale,
                scale,
                mean_constant_theta,
                preprocessing_value,
            )
            error = np.degrees(
                np.arccos(
                    np.dot(
                        vector_one(
                            predicted_vector_phi[0][0], predicted_vector_theta[0][0]
                        ),
                        vector_one(y_test_data_phi[k][0], y_test_data_theta[k][0]),
                    )
                )
            )
            error_distribution[k] += float(error)
    error_distribution = np.array(error_distribution)
    error_distribution /= float(n_point)

    with Path(output_path).open("w") as f:
        for i in range(len(error_distribution)):
            print(error_distribution[i], file=f)

    logger.info(f"Saved {output_path}")
