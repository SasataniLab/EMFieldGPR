"""Module for deciding 6 points in active learning."""

from pathlib import Path

import numpy as np
import torch

from EMFieldML.config import config, get_logger, paths, template
from EMFieldML.Learning import Learning

logger = get_logger(__name__)


class Decide6points:
    """Class for deciding 6 points in active learning process."""

    @staticmethod
    def get_pattern(
        i: int,
        j: int,
        k: int,
    ) -> int:
        """
        Divide the entire space into 27 blocks.
        i corresponds to x, j corresponds to y, and k corresponds to z.
        The if statements are complex to account for overlapping regions.

        Parameters
        ----------
        i : int
            i corresponds to x
        j : int
            j corresponds to y
        k : int
            k corresponds to z

        Returns
        -------
        int
            The index of the block in which the point (i, j, k) is located.

        """
        if k in {2, 3, 5, 6} and 2 <= i <= 7 and 2 <= j <= 7:
            return None

        x = (
            0
            if i <= 1 or (i == 2 and 2 <= j <= 6)
            else 2 if i >= 8 or (i == 7 and 2 <= j <= 6) else 1
        )
        y = (
            0
            if j <= 1 or (j == 2 and 3 <= i <= 6)
            else 2 if j >= 8 or (j == 7 and 2 <= i <= 7) else 1
        )
        z = 0 if k in {0, 1, 2} else 1 if k in {3, 4, 5} else 2

        return x + y * 3 + z * 9

    @staticmethod
    def make_pattern_list() -> None:
        """
        Assign each point to a block one by one.
        Exclude those that overlap with the ferrite shield.
        """
        number_list = [[] for _ in range(config.n_blocks)]
        count = 0

        for i in range(10):
            for j in range(10):
                for k in range(9):
                    index = Decide6points.get_pattern(i, j, k)
                    if index is not None:
                        number_list[index].append(count)
                        count += 1

        for pattern in range(config.n_blocks):
            output_path = paths.PATTERN_DIR / template.pattern_list.format(
                index=pattern
            )
            with Path(output_path).open("w") as f:
                for index_point in number_list[pattern]:
                    print(index_point, file=f)
            logger.info(f"Saved {output_path}")

    @staticmethod
    def make_learning_model_all() -> None:
        """Create learning models for all patterns."""
        input_path_train_list = paths.TRAIN_DIR / template.x_train_init
        x_train_data = Learning.make_x_train_data(input_path_train_list)

        for index in range(config.n_prediction_points_level1):
            input_path_y_data = (
                paths.Y_DATA_FOR_ACTIVE_LEARNING_DIR
                / template.y_data_magnitude.format(index=index)
            )
            output_path = (
                paths.LEARNING_MODEL_FOR_ACTIVE_LEARNING_DIR
                / template.model_point.format(index=index)
            )
            print(f"Start learning_magneticfield No.{index}")
            Learning.learning_magneticfield(
                input_path_y_data, output_path, x_train_data
            )

    @staticmethod
    def pre_deviation(
        input_path_y_data: Path,
        input_path_model: Path,
        x_train_data: np.ndarray,
        init_test_data_list: list[int],
    ):
        """Calculate pre-deviation for given data."""
        model, likelihood, _, _ = Learning.prepare_model(
            input_path_y_data, x_train_data
        )

        checkpoint = torch.load(input_path_model)
        model.load_state_dict(checkpoint["model_state_dict"])
        likelihood.load_state_dict(checkpoint["likelihood_state_dict"])

        # Find optimal model hyperparameters
        model.eval()
        likelihood.eval()
        deviation_average = 0

        for index_test in init_test_data_list:
            input_path_circle = paths.CIRCLE_MOVE_X_DIR / template.circle_move_x.format(
                index=index_test
            )
            deviation = Learning.calculate_deviation(
                input_path_circle, model, likelihood
            )
            deviation_average += float(deviation)

        deviation_average /= len(init_test_data_list)

        return deviation_average

    @staticmethod
    def predict_deviation_all() -> None:
        """Predict deviation for all patterns."""
        input_path_train_list = paths.TRAIN_DIR / template.x_train_init
        x_train_data = Learning.make_x_train_data(input_path_train_list)

        input_path_test_list = paths.TRAIN_DIR / template.x_test_init
        with Path(input_path_test_list).open() as f:
            init_test_data_list = [int(line.strip()) for line in f]

        for index in range(config.n_prediction_points_level1):
            input_path_y_data = (
                paths.Y_DATA_FOR_ACTIVE_LEARNING_DIR
                / template.y_data_magnitude.format(index=index)
            )
            input_path_model = (
                paths.LEARNING_MODEL_FOR_ACTIVE_LEARNING_DIR
                / template.model_point.format(index=index)
            )

            deviation = Decide6points.pre_deviation(
                input_path_y_data, input_path_model, x_train_data, init_test_data_list
            )

            output_path = (
                paths.DEVIATION_RESULT_FOR_ACTIVE_LEARNING_DIR
                / template.deviation_result.format(index=index)
            )
            with Path(output_path).open("w") as f:
                print(deviation, file=f)

            logger.info(f"Saved {output_path}")

    @staticmethod
    def check_deviation() -> None:
        """Check deviation values."""
        list_deviation = [[0, i, 0] for i in range(config.n_prediction_points_level1)]

        for index in range(config.n_blocks):
            input_path_pattern_list = paths.PATTERN_DIR / template.pattern_list.format(
                index=index
            )
            with Path(input_path_pattern_list).open() as f:
                for num in f:
                    list_deviation[int(num)][2] = index

        for index in range(config.n_prediction_points_level1):
            input_path_deviation = (
                paths.DEVIATION_RESULT_FOR_ACTIVE_LEARNING_DIR
                / template.deviation_result.format(index=index)
            )
            with Path(input_path_deviation).open() as f:
                for num in f:
                    list_deviation[index][0] = float(num)

        list_deviation.sort(key=lambda x: x[0])
        i = -1
        now_list = [0 for _ in range(config.n_blocks)]
        count = 0
        while True:
            if now_list[list_deviation[i][2]] == 0:
                print(
                    f"Selected point: {list_deviation[i][1]}  (Pattern: {list_deviation[i][2]}, Deviation: {list_deviation[i][0]})"
                )
                now_list[list_deviation[i][2]] = 1
                count += 1
            i -= 1
            if count == 6:
                break
