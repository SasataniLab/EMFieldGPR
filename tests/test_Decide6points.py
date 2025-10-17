import tempfile
from pathlib import Path

import pytest

from EMFieldML.ActiveLearning.Decide6points import Decide6points
from EMFieldML.config import paths, template


def test_make_pattern_list():
    """Test pattern list generation with real data validation

    Input: config parameters (spatial partitioning rules)
    Output: pattern_list files (pattern0_point_list.txt, pattern1_point_list.txt, etc.)
    """
    # Use real test data from tests/data
    test_data_dir = Path(__file__).parent / "data" / "active_learning" / "pattern_list"
    if not test_data_dir.exists():
        pytest.fail("Test data not available - pattern_list directory missing")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "pattern_list"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Mock the paths to use our test directory
        original_pattern_dir = paths.PATTERN_DIR
        paths.PATTERN_DIR = output_dir

        try:
            Decide6points.make_pattern_list()

            # Compare with real test data
            for i in range(3):
                generated_file = output_dir / template.pattern_list.format(index=i)
                expected_file = test_data_dir / template.pattern_list.format(index=i)

                assert generated_file.exists(), f"Pattern file {i} was not created"

                # Compare file contents exactly
                with open(generated_file) as f:
                    generated_content = f.read().strip()
                with open(expected_file) as f:
                    expected_content = f.read().strip()

                # Parse both files as sets of integers for comparison
                generated_points = {
                    int(line) for line in generated_content.split("\n") if line.strip()
                }
                expected_points = {
                    int(line) for line in expected_content.split("\n") if line.strip()
                }

                # Compare the actual point sets
                assert (
                    generated_points == expected_points
                ), f"Pattern {i} content mismatch:\nGenerated: {sorted(generated_points)}\nExpected: {sorted(expected_points)}"

                # Verify the points are valid (within expected range)
                assert all(
                    0 <= point < 900 for point in generated_points
                ), f"Pattern {i} contains out-of-range points: {generated_points}"

        finally:
            paths.PATTERN_DIR = original_pattern_dir


def test_get_pattern():
    """Test the get_pattern method for spatial partitioning

    Input: coordinate values (i, j, k)
    Output: pattern index (integer) or None for overlap regions
    """
    # Test various coordinate combinations
    assert Decide6points.get_pattern(0, 0, 0) == 0  # Corner case
    assert Decide6points.get_pattern(5, 5, 4) == 13  # Middle region
    assert Decide6points.get_pattern(9, 9, 8) == 26  # Another corner case
    assert Decide6points.get_pattern(4, 4, 4) == 13  # Middle region
