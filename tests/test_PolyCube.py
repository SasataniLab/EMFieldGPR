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
    """Test PolyCube mesh generation

    Input: circle_stl_{index}.stl (STL mesh files)
    Output: circle_x_data_{index}.txt (PolyCube mesh data)
    """
    float_assert_precision = 1e-3

    input_path = (
        paths.CI_TEST_DATA_DIR
        / "circle_stl"
        / template.circle_stl.format(index=data_index)
    )
    assert_file_path = (
        paths.CI_TEST_DATA_DIR / "circle_x" / template.circle_x.format(index=data_index)
    )

    # This with block ensures the temporary directory exists for the whole operation
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / template.circle_x.format(index=data_index)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        validator = FloatAssert(
            precision=float_assert_precision,
        )

        # Assuming PolyCubeMesher._make_grid writes to output_path.
        PolyCubeMaker.make_modeling(
            input_path=input_path,
            output_path_polycube=output_path,
            output_path_exterior=None,
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
            # Check that all lines contain valid numeric data
            for line_num, line in enumerate(lines):
                values = line.strip().split()
                assert len(values) > 0, f"Line {line_num} is empty"
                for value in values:
                    try:
                        float(value)
                    except ValueError:
                        pytest.fail(
                            f"Line {line_num} contains non-numeric value: {value}"
                        )
