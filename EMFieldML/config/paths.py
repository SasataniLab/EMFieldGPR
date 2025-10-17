"""Path configuration for EMFieldML electromagnetic field toolkit."""

from pathlib import Path

from .base import BaseConfig


class EMFieldMLPaths(BaseConfig):
    """Path configuration class for EMFieldML electromagnetic field toolkit."""

    def __init__(self) -> None:
        """Initialize EMFieldML paths configuration."""
        # Base directory - points to project root
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent

        # Data directory names
        self.Y_DATA = "y_data"
        self.TEST_Y_DATA = "test_y_data"

        # Core directories
        self.LUA_DIR = self.BASE_DIR / "EMFieldML" / "FEKO" / "Lua"
        self.DATA_DIR = self.BASE_DIR / "data"
        self.TEST_DATA_DIR = self.BASE_DIR / "test_data"
        self.CI_TEST_DATA_DIR = self.BASE_DIR / "tests" / "data"

        # Active learning directories
        self.ACTIVE_LEARNING_DIR = self.DATA_DIR / "active_learning"
        self.DEVIATION_RESULT_FOR_ACTIVE_LEARNING_DIR = (
            self.ACTIVE_LEARNING_DIR / "deviation_result"
        )
        self.PATTERN_DIR = self.ACTIVE_LEARNING_DIR / "pattern_list"
        self.Y_DATA_FOR_ACTIVE_LEARNING_DIR = self.ACTIVE_LEARNING_DIR / self.Y_DATA
        self.LEARNING_MODEL_FOR_ACTIVE_LEARNING_DIR = (
            self.ACTIVE_LEARNING_DIR / "learning_model"
        )
        self.LEARNING_MODEL_SELECTED_FOR_ACTIVE_LEARNING_DIR = (
            self.ACTIVE_LEARNING_DIR / "learning_model_selected_points"
        )

        # Circle data directories
        self.CIRCLE_DIR = self.DATA_DIR / "circle_parameter"
        self.CIRCLE_X_DIR = self.DATA_DIR / "circle_x_data"
        self.CIRCLE_MOVE_X_DIR = self.DATA_DIR / "circle_move_x_data"
        self.CIRCLE_STL_DIR = self.DATA_DIR / "stl"
        self.CIRCLE_PREDICTION_POINT_DIR = self.DATA_DIR / "circle_prediction_point"
        self.CIRCLE_FEKO_RAW_DIR = self.DATA_DIR / "circle_raw_data"

        # Training and model directories
        self.TRAIN_DIR = self.DATA_DIR / "train"
        self.LEARNING_MODEL_DIR = self.DATA_DIR / "learning_model"
        self.DEFORMATION_DIR = self.DATA_DIR / "deformation"

        # Y data directories
        self.Y_DATA_DIR = self.DATA_DIR / self.Y_DATA

        # Test data directories
        self.TEST_CIRCLE_MOVE = self.TEST_DATA_DIR / "circle_move_test"
        self.TEST_CIRCLE_FEKO_RAW_DIR = self.TEST_DATA_DIR / "circle_raw_data_test"
        self.TEST_CIRCLE_STL_DIR = self.TEST_DATA_DIR / "test_stl"
        self.TEST_CIRCLE_MOVE_X_DIR = self.TEST_DATA_DIR / "circle_move_test_x_data"
        self.TEST_CIRCLE_PREDICTION_POINT_DIR = (
            self.TEST_DATA_DIR / "circle_test_prediction_point"
        )
        self.TEST_Y_DATA_DIR = self.TEST_DATA_DIR / self.TEST_Y_DATA

        # Result directories
        self.RESULT_N_CHANGE_DIR = self.DATA_DIR / "result_n_change"
        self.RESULT_CROSS_DIR = self.DATA_DIR / "result_cross"
        self.RESULT_CROSS_PATTERN_DIR = self.RESULT_CROSS_DIR / "pattern_{index}"
        self.RESULT_FULL_DATA_DIR = self.DATA_DIR / "result_fulldata"
        self.RESULT_MAGNITUDE_MAP_DIR = self.DATA_DIR / "result_magnitude_map"
