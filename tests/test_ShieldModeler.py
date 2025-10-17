import tempfile
from pathlib import Path

import pytest

from EMFieldML.config import paths, template
from EMFieldML.Utils.FloatAssert import FloatAssert
from EMFieldML.Utils.ShieldModeler import ShieldModeler


def test_make_shield_shape():
    """Test shield parameter generation

    Input: config parameters (shield dimensions, radius ranges)
    Output: circle_param_{index}.txt files (shield parameter files)
    """
    float_assert_precision = 1e-3

    test_indices = [1, 41, 364]  # Arbitrary test indices to check

    # This with block ensures the temporary directory exists for the whole operation
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        validator = FloatAssert(
            precision=float_assert_precision,
        )

        ShieldModeler._make_shield_shape(
            output_dir=output_dir,
        )

        for test_index in test_indices:
            output_path = output_dir / template.circle_param.format(index=test_index)
            assert_file_path = (
                paths.CI_TEST_DATA_DIR
                / "circle_parameter"
                / template.circle_param.format(index=test_index)
            )

            assert (
                output_path.exists()
            ), f"Output file was not created for data_index {test_index}"
            assert validator.compare_files(
                output_path, assert_file_path
            ), f"Content mismatch for data_index {test_index}"

            # Additional validation: check parameter file has correct number of values
            with open(output_path) as f:
                content = f.read().strip()
                values = content.split()
                assert (
                    len(values) == 7
                ), f"Parameter file should have 7 values, got {len(values)}: {values}"
                for value in values:
                    try:
                        float(value)
                    except ValueError:
                        pytest.fail(
                            f"Parameter file contains non-numeric value: {value}"
                        )


def test_make_moved_shield_coordinate():
    """Test moved shield coordinate generation

    Input: circle_param_{index}.txt files (shield parameters)
    Output: circle_param_move_{index}.txt files (moved shield coordinates)
    """
    float_assert_precision = 1e-3

    test_indices = [1, 41, 364]  # Arbitrary test indices to check

    # This with block ensures the temporary directory exists for the whole operation
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        validator = FloatAssert(
            precision=float_assert_precision,
        )

        ShieldModeler._make_shield_shape(
            output_dir=output_dir,
        )

        ShieldModeler._make_moved_shield_coordinate(
            output_dir=output_dir,
        )

        for test_index in test_indices:
            output_path = output_dir / template.circle_move_param.format(
                index=test_index
            )
            assert_file_path = (
                paths.CI_TEST_DATA_DIR
                / "circle_parameter"
                / template.circle_move_param.format(index=test_index)
            )

            assert (
                output_path.exists()
            ), f"Output file was not created for data_index {test_index}"
            assert validator.compare_files(
                output_path, assert_file_path
            ), f"Content mismatch for data_index {test_index}"
