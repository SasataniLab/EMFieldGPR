"""Configuration management for EMFieldML electromagnetic field toolkit.

This module provides configuration classes and settings for the
electromagnetic field machine learning toolkit.
"""

import logging
from pathlib import Path

# Import logger directly to avoid circular dependency
try:
    from EMFieldML.Utils.logger import Logger
except ImportError:
    # Fallback to basic logging if EMFieldML package is not available
    class Logger:
        """Fallback logger class when EMFieldML package is not available."""

        def __init__(self, level=logging.INFO):
            """Initialize the fallback logger.

            Args:
                level: The logging level for the logger.

            """
            self.level = level

        def get_logger(self, name):
            """Create and configure a logger with the provided name.

            Args:
                name: The name for the logger.

            Returns:
                logging.Logger: Configured logger instance.

            """
            logger = logging.getLogger(name)
            logger.setLevel(self.level)
            return logger


class BaseConfig:
    """Base configuration class for EMFieldML electromagnetic field toolkit."""

    def dump_attrs(self):
        """
        Dump all attributes of the class.
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith("__")}

    def show_attrs(self):
        """
        Show all attributes of the class.
        """
        for k, v in self.dump_attrs().items():
            print(f"{k}: {v}")


class EMFieldMLConfig(BaseConfig):
    """Main configuration class for EMFieldML electromagnetic field toolkit."""

    def __init__(self):
        """Initialize EMFieldML configuration with default values.

        .. todo::
           Add figure references and equation numbers to Attributes section
           (See [Figure name][part] in README.md).

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
        self.n_shield_shape: int = 372
        self.n_initial_points: int = 72
        self.n_initial_train_data: int = 744
        self.n_train_data: int = 1488
        self.n_blocks = 27
        self.n_all_data: int = 62868  # 372 * 169
        self.dimension_x: int = 3368
        self.n_prediction_points_init: int = 900
        self.n_prediction_points_level1: int = 756
        self.n_prediction_points_level2: int = 5411
        self.n_prediction_points_level3: int = 40767
        self.n_active_learning_cycle: int = 124
        self.n_test_data = 50
        self.n_circle = 50
        self.prepocessing_value_magnitude: float = 20
        self.prepocessing_value_vector: float = 1
        self.prepocessing_value_efficiency: float = 1 / 6
        self.r1_min: int = 14
        self.r1_max: int = 21
        self.r3_min: int = 13
        self.r4_list: list[int] = [4, 7]
        self.radius_step: int = 2
        self.height_min: int = 1
        self.height_max: int = 4
        self.height_separate: float = 0.02
        self.meter_to_centimeter: float = 100
        self.convert_efficiency: float = 100
        self.scaler: float = 0.01
        self.y_grid: int = 13
        self.z_grid: int = 13
        self.y_step: float = 0.02
        self.z_step: float = 0.02
        self.initial_coil_distance: float = 0.05
        self.LBS_w_make: int = 1.5
        self.LBS_w_move: int = 5.0
        self.lua_trial: int = 5
        self.x_init_prediction_point_list = [
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
        self.y_init_prediction_point_list = [
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
        self.z_init_prediction_point_list = [
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
        self.y_data_x_point_num = 40
        self.y_data_y_point_num = 50
        self.y_data_z_point_num = 45
        self.y_data_x_point_list = [
            -0.40 + i * 0.80 / 39 for i in range(self.y_data_x_point_num)
        ]
        self.y_data_y_point_list = [
            -0.40 + i * 1.00 / 49 for i in range(self.y_data_y_point_num)
        ]
        self.y_data_z_point_list = [
            -0.20 + i * 0.70 / 44 for i in range(self.y_data_z_point_num)
        ]
        self.selected_points = [128, 210, 266, 276, 327, 540]
        self.n_cross_validation = 12
        self.log_level: int = logging.INFO


class EMFieldMLPaths(BaseConfig):
    """Path configuration class for EMFieldML electromagnetic field toolkit."""

    def __init__(self):
        """Initialize EMFieldML paths configuration."""
        self.BASE_DIR = Path(__file__).resolve().parent.parent
        self.Y_DATA = "y_data"
        self.TEST_Y_DATA = "test_y_data"
        self.LUA_DIR = self.BASE_DIR / "EMFieldML" / "FEKO" / "Lua"
        self.DATA_DIR = self.BASE_DIR / "data"
        self.TEST_DATA_DIR = self.BASE_DIR / "test_data"
        self.CI_TEST_DATA_DIR = self.BASE_DIR / "tests" / "data"
        self.ACTIVE_LEARNING_DIR = self.DATA_DIR / "active_learning"
        self.CIRCLE_DIR = self.DATA_DIR / "circle_parameter"
        self.CIRCLE_X_DIR = self.DATA_DIR / "circle_x_data"
        self.CIRCLE_MOVE_X_DIR = self.DATA_DIR / "circle_move_x_data"
        self.CIRCLE_STL_DIR = self.DATA_DIR / "stl"
        self.CIRCLE_PREDICTION_POINT_DIR = self.DATA_DIR / "circle_prediction_point"
        self.CIRCLE_FEKO_RAW_DIR = self.DATA_DIR / "circle_raw_data"
        self.DEVIATION_RESULT_FOR_ACTIVE_LEARNING_DIR = (
            self.ACTIVE_LEARNING_DIR / "deviation_result"
        )
        self.PATTERN_DIR = self.ACTIVE_LEARNING_DIR / "pattern_list"
        self.Y_DATA_DIR = self.DATA_DIR / self.Y_DATA
        self.Y_DATA_FOR_ACTIVE_LEARNING_DIR = self.ACTIVE_LEARNING_DIR / self.Y_DATA
        self.LEARNING_MODEL_FOR_ACTIVE_LEARNING_DIR = (
            self.ACTIVE_LEARNING_DIR / "learning_model"
        )
        self.LEARNING_MODEL_SELECTED_FOR_ACTIVE_LEARNING_DIR = (
            self.ACTIVE_LEARNING_DIR / "learning_model_selected_points"
        )
        self.TRAIN_DIR = self.DATA_DIR / "train"
        self.LEARNING_MODEL_DIR = self.DATA_DIR / "learning_model"
        self.DEFORMATION_DIR = self.DATA_DIR / "deformation"
        self.TEST_CIRCLE_MOVE = self.TEST_DATA_DIR / "circle_move_test"
        self.TEST_CIRCLE_FEKO_RAW_DIR = self.TEST_DATA_DIR / "circle_raw_data_test"
        self.TEST_CIRCLE_STL_DIR = self.TEST_DATA_DIR / "test_stl"
        self.TEST_CIRCLE_MOVE_X_DIR = self.TEST_DATA_DIR / "circle_move_test_x_data"
        self.TEST_CIRCLE_PREDICTION_POINT_DIR = (
            self.TEST_DATA_DIR / "circle_test_prediction_point"
        )
        self.TEST_Y_DATA_DIR = self.TEST_DATA_DIR / self.TEST_Y_DATA
        self.RESULT_N_CHANGE_DIR = self.DATA_DIR / "result_n_change"
        self.RESULT_CROSS_DIR = self.DATA_DIR / "result_cross"
        self.RESULT_CROSS_PATTERN_DIR = self.RESULT_CROSS_DIR / "pattern_{index}"


class EMFieldMLTemplate(BaseConfig):
    """Template configuration class for EMFieldML electromagnetic field toolkit."""

    def __init__(self):
        """Initialize EMFieldML template configuration."""
        self.circle_param: str = "circle_param_{index}.txt"
        self.circle_move_param: str = "circle_param_move_{index}.txt"
        self.circle_x: str = "circle_x_data_{index}.txt"
        self.circle_stl: str = "circle_stl_{index}.stl"
        self.circle_move_x: str = "ferrite_circle_move_{index}.txt"
        self.circle_prediction_point: str = "circle_prediction_point_{index}.txt"
        self.circle_prediction_point_move: str = (
            "circle_prediction_point_move_{index}.txt"
        )
        self.deviation_result: str = "deviation_result_{index}.txt"
        self.model_point: str = "model_point_{index}.pth"
        self.model: str = "model_efficiency.pth"
        self.x_train_init: str = "x_train_init.txt"
        self.x_train: str = "x_train.txt"
        self.x_train_after_active_learning: str = "x_train_after_active_learning.txt"
        self.x_test_init: str = "x_test_init.txt"
        self.x_test: str = "x_test.txt"
        self.x_test_after_active_learning: str = "x_test_after_active_learning.txt"
        self.pattern_list: str = "pattern{index}_point_list.txt"
        self.y_data_magnitude: str = "outdata_magnitude_{index}.csv"
        self.y_data_vector_theta: str = "outdata_vector_theta_{index}.csv"
        self.y_data_vector_phi: str = "outdata_vector_phi_{index}.csv"
        self.y_data_efficiency: str = "outdata_efficiency.csv"
        self.raw_data: str = "Circle_{index}_3.out"
        self.raw_data_sparameter: str = "Circle_{index}_2_SParameter1.s2p"
        self.raw_data_optimization: str = "Circle_{index}_optimum.out"
        self.test_circle_move_param: str = "circle_param_move_test_{index}.txt"
        self.test_circle_x: str = "circle_test_x_data_{index}.txt"
        self.test_circle_move: str = "ferrite_circle_move_test_{index}.txt"
        self.test_circle_stl: str = "circle_test_stl_{index}.stl"
        self.test_circle_prediction_point: str = (
            "circle_test_prediction_point_{index}.txt"
        )
        self.test_circle_prediction_point_move: str = (
            "circle_move_test_prediction_point_{index}.txt"
        )
        self.result_n_change_model: str = "model_point{index}_N_{n_data}.pth"
        self.result_n_change_deviation: str = (
            "point_number_deviation_{index}_N_{n_data}.txt"
        )
        self.result_n_change_error: str = "point_number_error_{index}_N_{n_data}.txt"
        self.result_n_change_deviation_sorted: str = (
            "point_number_deviation_{index}_sorted.txt"
        )
        self.result_n_change_error_sorted: str = "point_number_error_{index}_sorted.txt"


config = EMFieldMLConfig()
template = EMFieldMLTemplate()
paths = EMFieldMLPaths()
logger = Logger(level=config.log_level).get_logger("EMFieldML")
