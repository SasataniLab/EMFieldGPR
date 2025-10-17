"""Active learning module for electromagnetic field optimization.

This module implements active learning strategies to efficiently select
the most informative data points for training electromagnetic field
prediction models.
"""

from pathlib import Path

import numpy as np
import torch

from EMFieldML.config import config, paths, template
from EMFieldML.FEKO import FekoRunner
from EMFieldML.Learning.Learning import FieldGPRegressor
from EMFieldML.Learning.YDataMaker import TargetDataBuilder


class ActiveLearning:
    """Active learning strategies for electromagnetic field model optimization.

    This class implements methods to select the most informative data points
    for training, improving model performance with minimal data.
    """

    @staticmethod
    def pre_deviation(
        input_path_y_data: Path,
        input_path_model: Path,
        x_train_data: np.ndarray,
        test_data_list: list[int],
        point_index: int,
    ):
        """Calculate deviation for active learning point selection.

        Identify and select test data samples that show a large variance
        with respect to the current model's predictions.

        Args:
            input_path_y_data: Path to training target data
            input_path_model: Path to trained model file
            x_train_data: Training input features
            test_data_list: List of test data indices
            point_index: Index of prediction point

        Returns:
            Average and maximum deviation values

        Note:
            This function corresponds to 5 steps in Figure 5a of the paper.

        """
        model, likelihood, _, _ = FieldGPRegressor.prepare_model(
            input_path_y_data, x_train_data
        )

        checkpoint = torch.load(input_path_model)
        model.load_state_dict(checkpoint["model_state_dict"])
        likelihood.load_state_dict(checkpoint["likelihood_state_dict"])

        # Find optimal model hyperparameters
        model.eval()
        likelihood.eval()
        deviation_list = [
            [0, ii, point_index]
            for ii in range(config.n_all_data - len(test_data_list))
        ]

        for count, index_test in enumerate(test_data_list):
            input_path_circle = paths.CIRCLE_MOVE_X_DIR / template.circle_move_x.format(
                index=index_test
            )
            deviation = FieldGPRegressor.calculate_deviation(
                input_path_circle, model, likelihood
            )
            deviation_list[count][0] = deviation

        return deviation_list

    @staticmethod
    def active_learning(X_test_list, predict_point_list):
        """Perform active learning process.

        Active learning iteratively selects instances from the test set that
        have high prediction uncertainty for the current model. These difficult
        cases are then used to retrain the model. This active learning approach
        enables the efficient creation of a robust model, particularly in
        scenarios where generating a large and diverse training dataset is difficult.

        Args:
            X_test_list: List of test data points
            predict_point_list: List of points to predict

        Returns:
            List of selected data points for training

        Note:
            This function corresponds to Figure 5a in the paper.

        """
        deviation_all = []

        input_path_train_list = paths.TRAIN_DIR / template.x_train
        x_train_data = FieldGPRegressor.make_x_train_data(input_path_train_list)

        input_path_test_list = paths.TRAIN_DIR / template.x_test_init
        with input_path_test_list.open() as f:
            init_test_data_list = [int(line.strip()) for line in f]

        for index in predict_point_list:
            input_path_y_data = (
                paths.Y_DATA_FOR_ACTIVE_LEARNING_DIR
                / template.y_data_magnitude.format(index=index)
            )
            output_path = (
                paths.LEARNING_MODEL_SELECTED_FOR_ACTIVE_LEARNING_DIR
                / template.model_point.format(index=index)
            )
            FieldGPRegressor.learning_magneticfield(
                input_path_y_data,
                output_path,
                x_train_data,
                iter_record=2000,
                inter_record=800,
            )

            input_path_model = output_path
            deviation_list = FieldGPRegressor.pre_deviation(
                input_path_y_data, input_path_model, x_train_data, init_test_data_list
            )

            deviation_all += deviation_list

        deviation_all = sorted(deviation_all, reverse=True, key=lambda x: x[0])

        ans_num = []
        leave_predict_point = predict_point_list.copy()

        for i in range(len(deviation_all)):
            if (
                deviation_all[i][2] in leave_predict_point
                and deviation_all[i][1] not in ans_num
            ):  #
                leave_predict_point.remove(deviation_all[i][2])
                ans_num.append(deviation_all[i][1])

        for i in range(len(ans_num)):
            ans_num[i] = X_test_list[ans_num[i]]
        return ans_num

    @staticmethod
    def uncertainly_sampling():
        """Perform uncertainty sampling for active learning.

        Iteratively selects data points with high prediction uncertainty,
        runs simulations for these points, and updates the training dataset.

        Note:
            This function corresponds to step 6 in Figure 5a of the paper.

        """
        for num_cycle in range(config.n_active_learning_cycle):
            print(f"start {num_cycle+1} cycles")  #
            input_path_train_list = paths.TRAIN_DIR / template.x_train
            with input_path_train_list.open() as f:  #
                lines = f.readlines()
                x_train_list = [int(line.strip()) for line in lines]

            input_path_test_list = paths.TRAIN_DIR / template.x_test
            with input_path_test_list.open() as f:  #
                lines = f.readlines()
                x_test_list = [int(line.strip()) for line in lines]

            predict_point_list = config.selected_points

            new_x_train_list = FieldGPRegressor.active_learning(
                x_test_list, predict_point_list
            )

            FekoRunner.run_simulation_list(new_x_train_list)

            for i in new_x_train_list:
                x_train_list.append(i)
                x_test_list.remove(i)

            TargetDataBuilder.make_new_ydatafile(
                x_train_list,
                paths.Y_DATA_FOR_ACTIVE_LEARNING_DIR,
                prediction_point_list=predict_point_list,
            )

            with (paths.TRAIN_DIR / template.x_train).open("w") as f:  #
                for i in x_train_list:
                    print(i, file=f)

            with (paths.TRAIN_DIR / template.x_test).open("w") as f:  #
                for i in x_test_list:
                    print(i, file=f)
