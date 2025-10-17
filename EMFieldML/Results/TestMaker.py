"""Test data generation for electromagnetic field models.

This module provides functionality to generate test coordinates and
validation data for electromagnetic field prediction models.
"""

from pathlib import Path

import numpy as np

from EMFieldML.config import config, get_logger, paths, template

logger = get_logger(__name__)


class TestMaker:
    """Generate test coordinates and validation data for electromagnetic field models."""

    @staticmethod
    def make_test_shield_coordinate():
        """Generate test shield coordinates for model validation."""
        for index in range(1, config.n_test_data + 1):
            output_path = (
                paths.TEST_CIRCLE_MOVE_X_DIR
                / template.test_circle_move_param.format(index=index)
            )
            r1 = np.random.uniform(0.14, 0.20)
            r3 = np.random.uniform(0.13, r1 - 0.01)
            r2 = np.random.uniform(r3 + 0.01, r1)
            r4 = np.random.uniform(0.03, 0.06)
            h1 = np.random.uniform(0.01, 0.03)
            h2 = np.random.uniform(0.01, 0.03)
            y_move = np.random.uniform(0, 0.24)
            z_move = np.random.uniform(0, 0.24)
            with Path(output_path).open("w") as f:
                print(
                    f"{r1:.04f}",
                    f"{r2:.04f}",
                    f"{r3:.04f}",
                    f"{r4:.04f}",
                    f"{h1:.04f}",
                    f"{h2:.04f}",
                    f"{h2:.04f}",
                    f"{y_move:.04f}",
                    f"{z_move:.04f}",
                    file=f,
                )
            logger.info(f"Saved {output_path}")
