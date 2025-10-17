"""Template configuration for EMFieldML electromagnetic field toolkit."""

from .base import BaseConfig


class EMFieldMLTemplate(BaseConfig):
    """Template configuration class for EMFieldML electromagnetic field toolkit."""

    def __init__(self) -> None:
        """Initialize EMFieldML template configuration."""
        # Circle parameter templates
        self.circle_param: str = "circle_param_{index}.txt"
        self.circle_move_param: str = "circle_param_move_{index}.txt"
        self.circle_x: str = "circle_x_data_{index}.txt"
        self.circle_stl: str = "circle_stl_{index}.stl"
        self.circle_move_x: str = "ferrite_circle_move_{index}.txt"
        self.circle_prediction_point: str = "circle_prediction_point_{index}.txt"
        self.circle_prediction_point_move: str = (
            "circle_prediction_point_move_{index}.txt"
        )

        # Model and training templates
        self.deviation_result: str = "deviation_result_{index}.txt"
        self.model_point: str = "model_point_{index}.pth"
        self.model: str = "model_efficiency.pth"
        self.x_train_init: str = "x_train_init.txt"
        self.x_train: str = "x_train.txt"
        self.x_train_after_active_learning: str = "x_train_after_active_learning.txt"
        self.x_test_init: str = "x_test_init.txt"
        self.x_test: str = "x_test.txt"
        self.x_test_after_active_learning: str = "x_test_after_active_learning.txt"

        # Pattern and data templates
        self.pattern_list: str = "pattern{index}_point_list.txt"
        self.y_data_magnitude: str = "outdata_magnitude_{index}.csv"
        self.y_data_vector_theta: str = "outdata_vector_theta_{index}.csv"
        self.y_data_vector_phi: str = "outdata_vector_phi_{index}.csv"
        self.y_data_efficiency: str = "outdata_efficiency.csv"

        # Raw data templates
        self.raw_data: str = "Circle_{index}_3.out"
        self.raw_data_sparameter: str = "Circle_{index}_2_SParameter1.s2p"
        self.raw_data_optimization: str = "Circle_{index}_optimum.out"

        # Test data templates
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

        # Result templates
        self.result_n_change_model: str = "model_point{index}_N_{n_data}.pth"
        self.result_n_change_deviation: str = (
            "point_number_deviation_{index}_N_{n_data}.txt"
        )
        self.result_n_change_error: str = "point_number_error_{index}_N_{n_data}.txt"
        self.result_n_change_deviation_sorted: str = (
            "point_number_deviation_{index}_sorted.txt"
        )
        self.result_n_change_error_sorted: str = "point_number_error_{index}_sorted.txt"
        self.result_postprocessing_change_magnitude_error: str = (
            "magnitude_error_result_{index}.txt"
        )
        self.result_postprocessing_change_efficiency_error_under: str = (
            "efficiency_error_result_1_{index}.txt"
        )
        self.result_postprocessing_change_efficiency_error: str = (
            "efficiency_error_result_{index}.txt"
        )
        self.result_postprocessing_change_vector_error_under: str = (
            "vector_error_result_1_{index}.txt"
        )
        self.result_postprocessing_change_vector_error: str = (
            "vector_error_result_{index}.txt"
        )
        self.result_predict_magnitude_no_move: str = "predict_magnitude_no_move.txt"
        self.result_predict_magnitude_move: str = "predict_magnitude_move.txt"
        self.result_predict_vector_move: str = "predict_vector_move.txt"
        self.result_predict_vector_no_move: str = "predict_vector_no_move.txt"
