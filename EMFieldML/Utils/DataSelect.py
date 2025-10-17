"""Data selection utilities for electromagnetic field training.

This module provides functionality to select initial training data
from the available dataset. Selected data are output to X_train_new.txt.


"""

from pathlib import Path

from EMFieldML.config import config, get_logger, paths, template

logger = get_logger(__name__)


class DataSelect:
    """Select and manage initial training data for electromagnetic field models."""

    @staticmethod
    def select_init_data(
        n_total_shapes: int = config.n_shield_shape,
    ) -> list[int]:
        """Select initial training data using a systematic sampling pattern."""
        init_data_list = []

        y = 3
        x = 0

        for i in range(1, n_total_shapes + 1):
            init_data_list.append((x * 13 + y) * n_total_shapes + i)
            if y == 12:
                if x == 12:
                    y = 3
                    x = 0
                else:
                    y = 0
                    x += 3
            else:
                y += 3

        y = 9
        x = 6

        for i in range(1, 13):
            init_data_list.append((x * 13 + y) * n_total_shapes + i)
            if y == 12:
                if x == 12:
                    y = 3
                    x = 0
                else:
                    y = 0
                    x += 3
            else:
                y += 3

        y = 0
        x = 0

        for i in range(13, n_total_shapes + 1):
            if (i - 13) % 22 == 0 and i != 365:
                init_data_list.append(i)
            else:
                init_data_list.append((x * 13 + y) * n_total_shapes + i)
                if y == 12:
                    if x == 12:
                        y = 0
                        x = 0
                    else:
                        y = 0
                        x += 3
                else:
                    y += 3
        init_data_list.sort()
        return init_data_list

    @staticmethod
    def generate_x_train_init(
        output_path_init: Path = None,
        output_path: Path = None,
    ) -> None:
        """Generate initial training data files from selected data indices."""
        # Set defaults if not provided
        if output_path_init is None:
            output_path_init = paths.TRAIN_DIR / template.x_train_init
        if output_path is None:
            output_path = paths.TRAIN_DIR / template.x_train

        init_data_list = DataSelect.select_init_data()
        with Path(output_path_init).open("w") as f:
            for i in range(len(init_data_list)):
                print(init_data_list[i], file=f)

        with Path(output_path).open("w") as f:
            for i in range(len(init_data_list)):
                print(init_data_list[i], file=f)

        logger.info(f"Saved {output_path_init, output_path}")

    @staticmethod
    def generate_x_test_init(
        input_path: Path = None,
        output_path_init: Path = None,
        output_path: Path = None,
    ) -> None:
        """Generate test data files from training data indices."""
        # Set defaults if not provided
        if input_path is None:
            input_path = paths.TRAIN_DIR / template.x_train_init
        if output_path_init is None:
            output_path_init = paths.TRAIN_DIR / template.x_test_init
        if output_path is None:
            output_path = paths.TRAIN_DIR / template.x_test

        with Path(input_path).open() as f:
            init_train_data_list = [int(line.strip()) for line in f]

        init_test_data_list = []

        index = 0

        for i in range(1, config.n_all_data + 1):
            if (index >= len(init_train_data_list)) or (
                i != init_train_data_list[index]
            ):
                init_test_data_list.append(i)
            else:
                index += 1

        with Path(output_path_init).open("w") as f:
            for i in range(len(init_test_data_list)):
                print(init_test_data_list[i], file=f)

        with Path(output_path).open("w") as f:
            for i in range(len(init_test_data_list)):
                print(init_test_data_list[i], file=f)

        logger.info(f"Saved {output_path_init, output_path}")
