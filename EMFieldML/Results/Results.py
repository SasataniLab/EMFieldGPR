"""Results processing module for EMFieldML.

This module provides functionality for processing and analyzing results
from electromagnetic field machine learning models.
"""

import copy
import statistics
from datetime import UTC, datetime
from pathlib import Path

import gpytorch
import numpy as np
import torch

from EMFieldML.config import config, get_logger, paths, template
from EMFieldML.Learning import Learning

logger = get_logger(__name__)


class ResultMaker:
    """Class for processing and analyzing ML model results."""

    @staticmethod
    def pre_deviation(
        input_path_y_data: Path,
        input_path_model: Path,
        x_train_data: np.ndarray,
        test_data_list: list[int],
    ):
        """Calculate deviation statistics for test data using trained model."""
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
        deviation_max = 0

        for index_test in test_data_list:
            input_path_circle = paths.CIRCLE_MOVE_X_DIR / template.circle_move_x.format(
                index=index_test
            )
            deviation = Learning.calculate_deviation(
                input_path_circle, model, likelihood
            )
            deviation_max = max(deviation_max, float(deviation))
            deviation_average += float(deviation)

        deviation_average /= len(test_data_list)

        return deviation_average, deviation_max

    @staticmethod
    def pre_magneticfield(
        input_path_y_data: Path,
        input_path_model: Path,
        x_train_data: np.ndarray,
        x_test_data: np.ndarray,
        y_test_data: np.ndarray,
        test_data_list: list[int],
    ):
        """Calculate magnetic field predictions and statistics for test data."""
        model, likelihood, _, _ = Learning.prepare_model(
            input_path_y_data, x_train_data
        )

        checkpoint = torch.load(input_path_model)
        model.load_state_dict(checkpoint["model_state_dict"])
        likelihood.load_state_dict(checkpoint["likelihood_state_dict"])

        # Find optimal model hyperparameters
        model.eval()
        likelihood.eval()
        error_distribution = []

        for test_num in test_data_list:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_x = torch.from_numpy(x_test_data).clone().float()
                observed_pred = likelihood(model(test_x))

            test_magnitude = y_test_data[test_num - 1, 0]
            error = (
                observed_pred.mean.numpy()[test_num - 1] - test_magnitude
            ) / test_magnitude
            error_distribution.append(error)

        return statistics.mean(error_distribution), statistics.pstdev(
            error_distribution
        )

    @staticmethod
    def make_N_change_deviation(
        prediction_point_num: int,
    ):
        """Calculate deviation changes for different N values."""
        n_change = 6
        change_step = config.n_initial_train_data // n_change

        for n in range(n_change + 1):
            N_train = config.n_initial_train_data + change_step * n
            x_train_list = []
            x_test_list = list(range(1, config.n_all_data + 1))
            with (paths.TRAIN_DIR / template.x_train_after_active_learning).open() as f:
                data = f.readlines()
                for i in range(N_train):
                    x_train_list.append(int(data[i].rstrip("\n")))
                    x_test_list.remove(int(data[i].rstrip("\n")))

            x_train_data = np.empty((0, config.dimension_x), float)

            for i in range(N_train):
                with (
                    paths.CIRCLE_MOVE_X_DIR
                    / template.circle_move_x.format(index=x_train_list[i])
                ).open() as f:
                    data = f.readlines()
                    row_data = [[float(row.rstrip("\n")) for row in data]]
                    x_train_data = np.append(x_train_data, row_data, axis=0)

            output_path = (
                paths.LEARNING_MODEL_DIR
                / template.result_n_change_model.format(
                    index=prediction_point_num, n_data=N_train
                )
            )
            input_path_y_data = paths.Y_DATA_DIR / template.y_data_magnitude.format(
                index=prediction_point_num
            )

            Learning.FieldGPRegressor.learning_magneticfield(
                input_path_y_data=input_path_y_data,
                output_path=output_path,
                x_train_data=x_train_data,
                lr_record=0.1,
                iter_record=500,
                inter_record=500,
                small_record=5.0,
            )

            deviation, deviation_max = ResultMaker.pre_deviation(
                input_path_y_data=input_path_y_data,
                input_path_model=output_path,
                x_train_data=x_train_data,
                test_data_list=x_test_list,
            )

            with (
                paths.RESULT_N_CHANGE_DIR
                / template.result_n_change_deviation.format(
                    index=prediction_point_num, n_data=N_train
                )
            ).open("w") as f:
                print(deviation, deviation_max, file=f)

    @staticmethod
    def make_N_change_error(
        prediction_point_num: int,
    ):
        """Calculate error changes for different N values."""
        n_change = 6
        change_step = config.n_initial_train_data / n_change

        for n in range(n_change + 1):
            N_train = config.n_initial_train_data + change_step * n
            x_train_list = []
            x_test_list = list(range(1, config.n_test_data + 1))
            with (paths.TRAIN_DIR / template.x_train_after_active_learning).open() as f:
                data = f.readlines()
                for i in range(N_train):
                    x_train_list.append(int(data[i].rstrip("\n")))

            x_train_data = np.empty((0, config.dimension_x), float)
            x_test_data = np.empty((0, config.dimension_x), float)

            for i in range(N_train):
                with (
                    paths.CIRCLE_MOVE_X_DIR
                    / template.circle_move_x.format(index=x_train_list[i])
                ).open() as f:
                    data = f.readlines()
                    row_data = [[float(row.rstrip("\n")) for row in data]]
                    x_train_data = np.append(x_train_data, row_data, axis=0)

            for i in range(config.n_test_data):
                with (
                    paths.TEST_CIRCLE_MOVE_X_DIR
                    / template.test_circle_move.format(index=x_test_list[i])
                ).open() as f:
                    data = f.readlines()
                    row_data = [[float(row.rstrip("\n")) for row in data]]
                    x_test_data = np.append(x_test_data, row_data, axis=0)

            y_test_data = Learning.FieldGPRegressor.make_y_train_data(
                paths.TEST_Y_DATA_DIR
                / template.y_data_magnitude.format(prediction_point_num)
            )

            output_path = (
                paths.LEARNING_MODEL_DIR
                / template.result_n_change_model.format(
                    index=prediction_point_num, n_data=N_train
                )
            )
            input_path_y_data = paths.Y_DATA_DIR / template.y_data_magnitude.format(
                index=prediction_point_num
            )

            error_average, error_standard_variation = ResultMaker.pre_magneticfield(
                input_path_y_data=input_path_y_data,
                input_path_model=output_path,
                x_train_data=x_train_data,
                x_test_data=x_test_data,
                y_test_data=y_test_data,
                test_data_list=x_test_list,
            )

            with (
                paths.RESULT_N_CHANGE_DIR
                / template.result_n_change_error.format(
                    index=prediction_point_num, n_data=N_train
                )
            ).open("w") as f:
                print(error_average, error_standard_variation, file=f)

    @staticmethod
    def sort_N_change_data():
        """Sort N change data by deviation values."""
        for n in config.selected_points:

            deviation = [[0, 0] for _ in range(7)]
            for i in range(7):
                with (
                    paths.RESULT_N_CHANGE_DIR
                    / template.result_n_change_deviation.format(
                        index=n, n_data=124 * (i + 6)
                    )
                ).open() as f:
                    data = f.read()
                    data = data.split()
                    deviation[i][0] = data[0]
                    deviation[i][1] = data[1]

            with (
                paths.RESULT_N_CHANGE_DIR
                / template.result_n_change_deviation_sorted.format(index=n)
            ).open("w") as f:
                for i in range(7):
                    print(deviation[i][0], deviation[i][1], file=f)

            error = [[0, 0] for _ in range(7)]
            for i in range(7):
                with (
                    paths.RESULT_N_CHANGE_DIR
                    / template.result_n_change_error.format(
                        index=n, n_data=124 * (i + 6)
                    )
                ).open() as f:
                    data = f.read()
                    data = data.split()
                    error[i][0] = data[0]
                    error[i][1] = data[1]

            with (
                paths.RESULT_N_CHANGE_DIR
                / template.result_n_change_error_sorted.format(index=n)
            ).open("w") as f:
                for i in range(7):
                    print(error[i][0], error[i][1], file=f)

    @staticmethod
    def cross_validation_pattern():
        """Generate cross-validation patterns for model evaluation."""
        np.random.seed(seed=17)
        with (paths.TRAIN_DIR / template.x_train_after_active_learning).open() as f:
            lines = f.readlines()
            x_train_data = [int(line.strip()) for line in lines]
        np.random.shuffle(x_train_data)

        for i in range(config.n_cross_validation):
            test_list = []
            train_list = copy.deepcopy(x_train_data)
            for j in range(config.n_train_data // config.n_cross_validation):
                test_list.append(
                    x_train_data[
                        i * config.n_train_data // config.n_cross_validation + j
                    ]
                )
                train_list.remove(
                    x_train_data[
                        i * config.n_train_data // config.n_cross_validation + j
                    ]
                )
            train_list.sort()
            test_list.sort()
            with (
                Path(str(paths.RESULT_CROSS_PATTERN_DIR).format(index=i + 1))
                / template.x_test
            ).open("w") as f:
                for number in test_list:
                    print(number, file=f)
            with (
                Path(str(paths.RESULT_CROSS_PATTERN_DIR).format(index=i + 1))
                / template.x_train
            ).open("w") as f:
                for number in train_list:
                    print(number, file=f)

    @staticmethod
    def make_fix_prediction_point():
        """Create fixed prediction points for analysis."""
        x = config.x_init_prediction_point_list
        y = config.y_init_prediction_point_list
        z = config.z_init_prediction_point_list

        with (
            paths.CIRCLE_PREDICTION_POINT_DIR
            / template.result_prediction_point_move_fix
        ).open("w") as f:
            for i in range(len(x)):
                for j in range(len(y)):
                    for k in range(len(z)):
                        print(x[i], y[j], z[k], file=f)

    @staticmethod
    def sort_list(
        input_path_dir: Path,
        input_path_file: str,
        output_path: Path,
    ):
        """Sort a list of data by the first element of each sublist."""
        error_list = []
        for i in range(1, config.n_cross_validation + 1):
            with (
                Path(str(input_path_dir).format(index=i)) / input_path_file
            ).open() as f:
                for num in f:
                    error_list.append(float(num))
        with Path(output_path).open("w") as f:
            for i in error_list:
                print(i, file=f)

        logger.info(f"Saved {output_path}")

    @staticmethod
    def check_time(file_path):
        """Check and parse time information from file."""
        with Path(file_path).open() as file:
            lines = file.readlines()
        lines_strip_time = [line.strip() for line in lines]
        list_rownum_time = [
            i for i, line_s in enumerate(lines_strip_time) if "Date:" in line_s
        ]
        line_split_time = lines_strip_time[list_rownum_time[0]].split()
        start_day_time = datetime.strptime(
            f"{line_split_time[1]} {line_split_time[2]}", "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=UTC)

        lines_strip_time = [line.strip() for line in lines]
        list_rownum_time = [
            i for i, line_s in enumerate(lines_strip_time) if "Finished:" in line_s
        ]
        line_split_time = lines_strip_time[list_rownum_time[0]].split()
        finish_day_time = datetime.strptime(
            f"{line_split_time[1]} {line_split_time[2]}", "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=UTC)

        delta = finish_day_time - start_day_time

        return delta.seconds

    @staticmethod
    def measure_simulation_time():
        """Measure and record simulation execution time."""
        train_list = []  #
        with (paths.TRAIN_DIR / template.x_train_after_active_learning).open() as f:
            for num in f:
                train_list.append(int(num))

        ans = 0
        for index in train_list:
            delta = ResultMaker.check_time(
                paths.CIRCLE_FEKO_RAW_DIR / "Circle_2" / f"Circle_{index}_2.out"
            )
            ans += delta
        print(f"2.out : {ans / len(list)} s")

        ans = 0
        for index in train_list:
            delta = ResultMaker.check_time(
                paths.CIRCLE_FEKO_RAW_DIR / f"Circle_{index}_3.out"
            )
            ans += delta
        print(f"3.out : {ans / len(list)} s")

        ans = 0
        folder_path = paths.CIRCLE_FEKO_RAW_DIR / "circle_optimum"
        # .outファイルをすべて取得
        out_files = list(folder_path.glob("*.out"))
        # 各ファイルにアクセス
        for file_path in out_files:
            delta = ResultMaker.check_time(file_path)
            ans += delta
        print(f"Optimum : {ans / len(list)} s")
