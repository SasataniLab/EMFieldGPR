"""Settings configuration for EMFieldML electromagnetic field toolkit."""

import logging
from typing import List

from .base import BaseConfig


class EMFieldMLConfig(BaseConfig):
    """Main configuration class for EMFieldML electromagnetic field toolkit."""

    def __init__(self) -> None:
        """Initialize EMFieldML configuration with default values.

        Attributes:
            n_shield_shape: Number of shield shapes
            n_initial_points: Number of initial points
            n_initial_train_data: Number of initial train data
            n_train_data: Number of train data
            n_blocks: Number of blocks
            n_all_data: Number of all data
            dimension_x: Dimension x
            n_prediction_points_init: Number of prediction points init
            n_prediction_points_level1: Number of prediction points level1
            n_prediction_points_level2: Number of prediction points level2
            n_prediction_points_level3: Number of prediction points level3
            n_active_learning_cycle: Number of active learning cycle
            n_test_data: Number of test data
            n_circle: Number of circle
            prepocessing_value_magnitude: Preprocessing value magnitude
            prepocessing_value_vector: Preprocessing value vector
            prepocessing_value_efficiency: Preprocessing value efficiency
            r1_min: Minimum r1
            r1_max: Maximum r1
            r3_min: Minimum r3
            r4_list: List of r4
            radius_step: Radius step
            height_min: Minimum height
            height_max: Maximum height
            height_separate: Height separate
            meter_to_centimeter: Meter to centimeter
            convert_efficiency: Convert efficiency
            scaler: Scaler
            y_grid: Y grid
            z_grid: Z grid
            y_step: Y step
            z_step: Z step
            initial_coil_distance: Initial coil distance
            LBS_w_make: LBS w make
            LBS_w_move: LBS w move
            lua_trial: Lua trial
            x_init_prediction_point_list: List of x init prediction point
            y_init_prediction_point_list: List of y init prediction point
            z_init_prediction_point_list: List of z init prediction point
            y_data_x_point_num: Number of y data x point
            y_data_y_point_num: Number of y data y point
            y_data_z_point_num: Number of y data z point
            y_data_x_point_list: List of y data x point
            y_data_y_point_list: List of y data y point
            y_data_z_point_list: List of y data z point
            selected_points: List of selected points
            n_cross_validation: Number of cross validation
            log_level: Log level

        """
        # Core parameters
        self.n_shield_shape: int = 372
        self.n_initial_points: int = 72
        self.n_initial_train_data: int = 744
        self.n_train_data: int = 1488
        self.n_blocks: int = 27
        self.n_all_data: int = 62868  # 372 * 169
        self.dimension_x: int = 3368

        # Prediction points
        self.n_prediction_points_init: int = 900
        self.n_prediction_points_level1: int = 756
        self.n_prediction_points_level2: int = 5411
        self.n_prediction_points_level3: int = 40767

        # Active learning
        self.n_active_learning_cycle: int = 124
        self.n_test_data: int = 50
        self.n_circle: int = 50

        # Preprocessing values
        self.prepocessing_value_magnitude: float = 20
        self.prepocessing_value_vector: float = 1
        self.prepocessing_value_efficiency: float = 1 / 6

        # Geometry parameters
        self.r1_min: int = 14
        self.r1_max: int = 21
        self.r3_min: int = 13
        self.r4_list: List[int] = [4, 7]
        self.radius_step: int = 2
        self.height_min: int = 1
        self.height_max: int = 4
        self.height_separate: float = 0.02

        # Conversion factors
        self.meter_to_centimeter: float = 100
        self.convert_efficiency: float = 100
        self.scaler: float = 0.01

        # Grid parameters
        self.y_grid: int = 13
        self.z_grid: int = 13
        self.y_step: float = 0.02
        self.z_step: float = 0.02

        # Coil parameters
        self.initial_coil_distance: float = 0.05
        self.LBS_w_make: int = 1.5
        self.LBS_w_move: int = 5.0
        self.lua_trial: int = 5

        # Prediction point lists
        self.x_init_prediction_point_list: List[float] = [
            -0.40,
            -0.325,
            -0.25,
            -0.15,
            -0.05,
            0.05,
            0.15,
            0.25,
            0.325,
            0.40,
        ]
        self.y_init_prediction_point_list: List[float] = [
            -0.40,
            -0.325,
            -0.25,
            -0.15,
            -0.05,
            0.05,
            0.15,
            0.25,
            0.325,
            0.40,
        ]
        self.z_init_prediction_point_list: List[float] = [
            -0.20,
            -0.13,
            -0.06,
            0.0,
            0.025,
            0.05,
            0.11,
            0.205,
            0.3,
        ]

        # Data point parameters
        self.y_data_x_point_num: int = 40
        self.y_data_y_point_num: int = 50
        self.y_data_z_point_num: int = 45

        # Generated point lists
        self.y_data_x_point_list: List[float] = [
            -0.40 + i * 0.80 / 39 for i in range(self.y_data_x_point_num)
        ]
        self.y_data_y_point_list: List[float] = [
            -0.40 + i * 1.00 / 49 for i in range(self.y_data_y_point_num)
        ]
        self.y_data_z_point_list: List[float] = [
            -0.20 + i * 0.70 / 44 for i in range(self.y_data_z_point_num)
        ]

        # Selected points and validation
        self.selected_points: List[int] = [128, 210, 266, 276, 327, 540]
        self.n_cross_validation: int = 12

        # Logging
        self.log_level: int = logging.INFO
