import tempfile
from pathlib import Path

import pytest

from EMFieldML.config import paths, template
from EMFieldML.Modeling.PolyCubeMaker import PolyCubeMaker
from EMFieldML.Utils.FloatAssert import FloatAssert


@pytest.mark.parametrize(
    "data_index",
    [
        1,
        41,
        364,
    ],
)
def test_make_grid(data_index: int):
    """Test exterior grid generation

    Input: circle_stl_{index}.stl (STL mesh files)
    Output: circle_prediction_point_{index}.txt (exterior grid points)
    """
    float_assert_precision = 1e-3

    input_path = (
        paths.CI_TEST_DATA_DIR
        / "circle_stl"
        / template.circle_stl.format(index=data_index)
    )
    assert_file_path = (
        paths.CI_TEST_DATA_DIR
        / "circle_prediction_point"
        / template.circle_prediction_point.format(index=data_index)
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / template.circle_prediction_point.format(
            index=data_index
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        validator = FloatAssert(
            precision=float_assert_precision,
        )

        PolyCubeMaker.make_modeling(
            input_path=input_path,
            output_path_polycube=None,
            output_path_exterior=output_path,
        )

        assert (
            output_path.exists()
        ), f"Output file was not created for data_index {data_index}"

        assert validator.compare_files(
            output_path, assert_file_path
        ), f"Content mismatch for data_index {data_index}"

        # Additional validation: check file has reasonable content
        with open(output_path) as f:
            lines = f.readlines()
            assert len(lines) > 0, f"Output file is empty for data_index {data_index}"
            # Check that all lines contain valid coordinate data (3 numbers per line)
            for line_num, line in enumerate(lines):
                coords = line.strip().split()
                assert (
                    len(coords) == 3
                ), f"Line {line_num} has wrong number of coordinates: {coords}"
                for coord in coords:
                    try:
                        float(coord)
                    except ValueError:
                        pytest.fail(
                            f"Line {line_num} contains non-numeric coordinate: {coord}"
                        )
