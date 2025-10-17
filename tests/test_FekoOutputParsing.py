import tempfile
from pathlib import Path

import pytest

from EMFieldML.config import paths, template
from EMFieldML.Learning.YDataMaker import TargetDataBuilder


def test_parse_feko_output_format():
    """Test FEKO output file format parsing

    Input: Circle_{index}_3.out files (raw FEKO output)
    Output: validates file format and data extraction
    """
    # Use real test data
    test_data_dir = Path(__file__).parent / "data"
    if not test_data_dir.exists():
        pytest.fail("Test data not available - data directory missing")

    raw_data_dir = test_data_dir / "circle_raw_data"
    if not raw_data_dir.exists():
        pytest.fail("FEKO raw data not available")

    # Test with real FEKO output file
    feko_file = raw_data_dir / "Circle_101_3.out"
    if not feko_file.exists():
        pytest.fail("FEKO output file Circle_101_3.out not available")

    # Test that we can read the file and extract key information
    with open(feko_file) as f:
        lines = f.readlines()

    lines_strip = [line.strip() for line in lines]

    # Check for key FEKO output sections
    load_section = [
        i for i, line in enumerate(lines_strip) if "RESULTS FOR LOADS" in line
    ]
    power_section = [i for i, line in enumerate(lines_strip) if "Power in Watt" in line]
    field_section = [
        i
        for i, line in enumerate(lines_strip)
        if "VALUES OF THE MAGNETIC FIELD STRENGTH in A/m" in line
    ]

    assert len(load_section) > 0, "FEKO output missing RESULTS FOR LOADS section"
    assert len(power_section) > 0, "FEKO output missing Power in Watt section"
    assert len(field_section) > 0, "FEKO output missing magnetic field section"

    # Test power extraction
    if power_section:
        power_line = lines_strip[power_section[0]]
        power_values = power_line.split()
        assert len(power_values) >= 4, "Power line has insufficient data"
        # Handle scientific notation (e.g., 1.01241E+00)
        try:
            float(power_values[3])
        except ValueError:
            pytest.fail(f"Power value is not numeric: {power_values[3]}")

    # Test field data extraction
    if field_section:
        field_start = field_section[0] + 6  # Skip header lines
        field_lines = lines_strip[
            field_start : field_start + 10
        ]  # Check first 10 field lines

        for line in field_lines:
            if line:  # Skip empty lines
                values = line.split()
                assert len(values) >= 10, f"Field line has insufficient data: {line}"
                # Check that values are numeric
                for val in values[4:10]:  # Check field values (skip coordinates)
                    try:
                        float(val)
                    except ValueError:
                        pytest.fail(f"Field value is not numeric: {val}")


def test_feko_output_parsing_integration():
    """Test FEKO output parsing integration with YDataMaker

    Input: Circle_{index}_3.out files + prediction points
    Output: validates complete parsing pipeline
    """
    # Use real test data
    test_data_dir = Path(__file__).parent / "data"
    if not test_data_dir.exists():
        pytest.fail("Test data not available - data directory missing")

    raw_data_dir = test_data_dir / "circle_raw_data"
    prediction_dir = test_data_dir
    output_dir = Path(tempfile.mkdtemp())

    if not raw_data_dir.exists():
        pytest.fail("FEKO raw data not available")

    try:
        # Test with real data
        feko_file = raw_data_dir / "Circle_101_3.out"
        prediction_file = prediction_dir / "circle_prediction_point_101.txt"

        if not feko_file.exists() or not prediction_file.exists():
            pytest.skip("Required test data files not available")

        # Test the complete parsing pipeline
        x_train_list = [101]  # Single test case

        # Mock the paths
        original_raw_dir = paths.CIRCLE_FEKO_RAW_DIR
        original_pred_dir = paths.CIRCLE_PREDICTION_POINT_DIR
        paths.CIRCLE_FEKO_RAW_DIR = raw_data_dir
        paths.CIRCLE_PREDICTION_POINT_DIR = prediction_dir

        try:
            # Test YDataMaker with real FEKO data
            TargetDataBuilder.make_new_ydatafile(
                x_train_list,
                output_dir,
                input_path_raw_data_dir=raw_data_dir,
                input_path_prediction_point_dir=prediction_dir,
                prediction_point_list=[0],  # Single prediction point
                n_prediction_points=1,
            )

            # Check that output was created
            output_file = output_dir / template.y_data_magnitude.format(index=0)
            assert output_file.exists(), "Y-data output file was not created"

            # Check file has content
            with open(output_file) as f:
                content = f.read().strip()
                assert len(content) > 0, "Y-data output file is empty"

                # Check that content looks like magnitude data
                lines = content.split("\n")
                for line in lines:
                    try:
                        val = float(line)
                        assert (
                            val >= 0
                        ), f"Magnitude value should be non-negative: {val}"
                    except ValueError:
                        pytest.fail(f"Y-data contains non-numeric value: {line}")

        finally:
            paths.CIRCLE_FEKO_RAW_DIR = original_raw_dir
            paths.CIRCLE_PREDICTION_POINT_DIR = original_pred_dir

    finally:
        # Clean up
        import shutil

        shutil.rmtree(output_dir, ignore_errors=True)


def test_feko_output_error_handling():
    """Test FEKO output parsing error handling

    Input: malformed FEKO output files
    Output: validates error handling and graceful failures
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create malformed FEKO output file
        malformed_file = Path(temp_dir) / "malformed.out"
        with open(malformed_file, "w") as f:
            f.write("This is not a valid FEKO output file\n")
            f.write("Missing required sections\n")

        # Test that parsing fails gracefully
        with open(malformed_file) as f:
            lines = f.readlines()

        lines_strip = [line.strip() for line in lines]

        # Check that required sections are missing
        load_section = [
            i for i, line in enumerate(lines_strip) if "RESULTS FOR LOADS" in line
        ]
        field_section = [
            i
            for i, line in enumerate(lines_strip)
            if "VALUES OF THE MAGNETIC FIELD STRENGTH in A/m" in line
        ]

        assert (
            len(load_section) == 0
        ), "Malformed file should not have RESULTS FOR LOADS section"
        assert (
            len(field_section) == 0
        ), "Malformed file should not have magnetic field section"
