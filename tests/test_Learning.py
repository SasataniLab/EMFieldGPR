"""Test module for Learning functionality.

This module contains tests for the Learning class and related functionality
in the EMFieldML package.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from EMFieldML.config import template
from EMFieldML.Learning.Learning import FieldGPRegressor


def test_make_x_train_data():
    """Test X training data loading with realistic data validation.

    This test validates the make_x_train_data function by creating synthetic
    electromagnetic field data that mimics the expected structure and range
    of real EM field measurements.

    Input: x_train.txt (list of data indices), circle_move_x_data files (text format)
    Output: numpy array (samples x 3368 features)

    Note: The 3368 feature count is based on the expected EM field data structure.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        train_list_file = Path(temp_dir) / "x_train.txt"
        circle_data_dir = Path(temp_dir) / "circle_data"
        circle_data_dir.mkdir(parents=True, exist_ok=True)

        # Create training list file with test data indices
        # These indices (101, 102, 103) are arbitrary test values.
        with train_list_file.open("w") as f:
            f.write("101\n102\n103\n")

        # Create circle data files with realistic data structure
        for i in [101, 102, 103]:
            circle_file = circle_data_dir / template.circle_move_x.format(index=i)
            with circle_file.open("w") as f:
                # Create realistic data with proper structure (like real EM field data)
                for j in range(3368):
                    # Simulate realistic EM field values with some variation
                    # The coefficients (0.01, 0.1, 0.005, 0.05, 0.0001) are
                    value = (
                        np.sin(j * 0.01) * 0.1 + np.cos(j * 0.005) * 0.05 + j * 0.0001
                    )
                    f.write(f"{value:.6f}\n")

        # Test the function
        result = FieldGPRegressor.make_x_train_data(
            train_list_file,
            str(circle_data_dir / template.circle_move_x),
        )

        # Check result
        assert result.shape[0] == 3  # 3 samples
        assert result.shape[1] == 3368  # 3368 features per sample
        assert not (result == 0).all(), "All values are zero - data loading failed"
        assert not (
            result == result[0, 0]
        ).all(), "All values are identical - data loading failed"

        # Verify data range is reasonable for EM field data
        # The threshold values (1.0 for max magnitude, 0.01 for std deviation)
        # are based on assumptions about typical EM field data ranges.
        assert np.all(
            np.abs(result) < 1.0
        ), f"Data values too large: max={np.max(np.abs(result))}"
        assert np.std(result) > 0.01, "Data has insufficient variation"


def test_make_y_train_data():
    """Test Y training data loading.

    This test validates the make_y_train_data function by creating synthetic
    target values that represent the expected output format for the ML model.

    Input: CSV file (y_data.csv with magnitude values)
    Output: numpy array (samples x 1 output)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test Y data file
        # The values (0.1, 0.2, 0.3) are arbitrary test values.
        # These should represent specific magnitude ranges.
        y_data_file = Path(temp_dir) / "y_data.csv"
        with y_data_file.open("w") as f:
            f.write("0.1\n0.2\n0.3\n")

        # Test the function
        result = FieldGPRegressor.make_y_train_data(str(y_data_file))

        # Check result
        assert result.shape[0] == 3  # 3 samples
        assert result.shape[1] == 1  # 1 output per sample


def test_make_x_train_data_with_real_data():
    """Test X training data loading with real data validation.

    This test validates the make_x_train_data function using actual test data
    when available, falling back to synthetic data if real data is not present.
    This ensures the function works with both real and synthetic data.

    Input: x_train.txt (list of data indices), ferrite_circle_move_1.txt (real data)
    Output: numpy array (samples x 3368 features) with real data validation
    """
    # Use real test data
    test_data_dir = Path(__file__).parent / "data"
    if not test_data_dir.exists():
        pytest.fail("Test data not available - data directory missing")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create training list file with real data indices
        # The index 101 is used to match the expected real data file.
        train_list_file = Path(temp_dir) / "x_train.txt"
        with train_list_file.open("w") as f:
            f.write("101\n")  # Use real data index

        # Copy real circle data
        circle_data_dir = Path(temp_dir) / "circle_data"
        circle_data_dir.mkdir(parents=True, exist_ok=True)

        # Use real circle data if available
        real_circle_file = (
            test_data_dir / "circle_move_x_data" / "ferrite_circle_move_1.txt"
        )
        if real_circle_file.exists():
            shutil.copy(
                real_circle_file,
                circle_data_dir / template.circle_move_x.format(index=101),
            )
        else:
            # Fallback to synthetic data
            circle_file = circle_data_dir / template.circle_move_x.format(index=101)
            with circle_file.open("w") as f:
                for j in range(3368):
                    f.write(f"{j * 0.001}\n")

        # Test the function
        result = FieldGPRegressor.make_x_train_data(
            train_list_file,
            str(circle_data_dir / template.circle_move_x),
        )

        # Validate result structure and content
        assert result.shape[0] == 1  # 1 sample
        assert result.shape[1] == 3368  # 3368 features per sample
        assert not (result == 0).all(), "All values are zero - data loading failed"
        assert not (
            result == result[0, 0]
        ).all(), "All values are identical - data loading failed"
