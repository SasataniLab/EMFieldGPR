import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from EMFieldML.config import paths, template
from EMFieldML.FEKO.FEKORunner import FekoRunner


def test_make_stl_list():
    """Test STL file generation for FEKO

    Input: circle_param_{index}.txt files (shield parameters)
    Output: circle_stl_{index}.stl files (STL mesh files)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test directories
        param_dir = Path(temp_dir) / "circle_parameter"
        stl_dir = Path(temp_dir) / "stl"
        param_dir.mkdir(parents=True, exist_ok=True)
        stl_dir.mkdir(parents=True, exist_ok=True)

        # Create test parameter files
        test_indices = [1, 2, 3]
        for i in test_indices:
            param_file = param_dir / template.circle_param.format(index=i)
            with open(param_file, "w") as f:
                f.write("1.0 2.0 3.0 4.0 5.0 6.0 7.0\n")  # Sample parameters

        # Mock the paths
        original_param_dir = paths.CIRCLE_DIR
        original_stl_dir = paths.CIRCLE_STL_DIR
        paths.CIRCLE_DIR = param_dir
        paths.CIRCLE_STL_DIR = stl_dir

        try:
            # Mock FEKO execution (since we don't have FEKO license)
            with patch(
                "EMFieldML.FEKO.FEKORunner.FekoRunner.run_feko"
            ) as mock_run_feko:

                def mock_run_feko_side_effect(*args, **kwargs):
                    # Create a dummy STL file when FEKO is "run"
                    # args[1] is input_data string containing the index
                    input_data = args[1]
                    # Extract index from input_data (format: "param_file\nstl_file")
                    stl_file_path = input_data.split("\n")[1]
                    stl_file = Path(stl_file_path)
                    stl_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(stl_file, "w") as f:
                        f.write("solid test\nendsolid test\n")
                    print(f"Created STL file: {stl_file}")  # Debug output

                mock_run_feko.side_effect = mock_run_feko_side_effect

                # Call make_stl_list with explicit path parameters
                for i in range(3):
                    FekoRunner.make_stl(
                        n_shield_shape=i,
                        path_circle_dir=param_dir,
                        path_save_dir=stl_dir,
                    )

                # Check that STL files were created
                for i in test_indices:
                    stl_file = stl_dir / template.circle_stl.format(index=i)
                    assert stl_file.exists(), f"STL file {i} was not created"

                    # Check file has content (even if mocked)
                    assert stl_file.stat().st_size > 0, f"STL file {i} is empty"

        finally:
            paths.CIRCLE_DIR = original_param_dir
            paths.CIRCLE_STL_DIR = original_stl_dir


def test_make_coil_stl():
    """Test coil STL file generation

    Input: config parameters (coil geometry)
    Output: coil.stl file (coil mesh file)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        stl_dir = Path(temp_dir) / "stl"
        stl_dir.mkdir(parents=True, exist_ok=True)

        # Mock the paths
        original_stl_dir = paths.CIRCLE_STL_DIR
        paths.CIRCLE_STL_DIR = stl_dir

        try:
            # Mock FEKO execution
            with patch(
                "EMFieldML.FEKO.FEKORunner.FekoRunner.run_feko"
            ) as mock_run_feko:

                def mock_run_feko_side_effect(*args, **kwargs):
                    # Create a dummy coil STL file when FEKO is "run"
                    coil_file = stl_dir / "coil.stl"
                    with open(coil_file, "w") as f:
                        f.write("solid coil\nendsolid coil\n")

                mock_run_feko.side_effect = mock_run_feko_side_effect

                FekoRunner.make_coil_stl()

                # Check that coil STL file was created
                coil_file = stl_dir / "coil.stl"
                assert coil_file.exists(), "Coil STL file was not created"

        finally:
            paths.CIRCLE_STL_DIR = original_stl_dir


def test_run_simulation_list():
    """Test FEKO simulation execution

    Input: list of data indices, STL files
    Output: Circle_{index}_3.out files (FEKO simulation results)
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test directories
        stl_dir = Path(temp_dir) / "stl"
        raw_data_dir = Path(temp_dir) / "raw_data"
        stl_dir.mkdir(parents=True, exist_ok=True)
        raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Create test STL files
        test_indices = [101, 102]
        for i in test_indices:
            stl_file = stl_dir / template.circle_stl.format(index=i)
            with open(stl_file, "w") as f:
                f.write("solid test\nendsolid test\n")  # Minimal STL content

        # Mock the paths
        original_stl_dir = paths.CIRCLE_STL_DIR
        original_raw_dir = paths.CIRCLE_FEKO_RAW_DIR
        paths.CIRCLE_STL_DIR = stl_dir
        paths.CIRCLE_FEKO_RAW_DIR = raw_data_dir

        try:
            # Mock FEKO execution
            with (
                patch(
                    "EMFieldML.FEKO.FEKORunner.FekoRunner.run_solver"
                ) as mock_run_solver,
                patch(
                    "EMFieldML.FEKO.FEKORunner.FekoRunner.delete_warning"
                ) as mock_delete_warning,
            ):

                def mock_run_solver_side_effect(index):
                    # Create dummy FEKO output files when solver is "run"
                    output_file = raw_data_dir / template.raw_data.format(index=index)
                    with open(output_file, "w") as f:
                        f.write("FEKO simulation output\n")
                        f.write("RESULTS FOR LOADS\n")
                        f.write("Power in Watt: 1.0\n")
                        f.write("VALUES OF THE MAGNETIC FIELD STRENGTH in A/m\n")
                        f.write("1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0\n")

                mock_run_solver.side_effect = mock_run_solver_side_effect
                mock_delete_warning.return_value = None

                FekoRunner.run_simulation_list(test_indices)

                # Check that FEKO output files were created
                for i in test_indices:
                    output_file = raw_data_dir / template.raw_data.format(index=i)
                    assert output_file.exists(), f"FEKO output file {i} was not created"

        finally:
            paths.CIRCLE_STL_DIR = original_stl_dir
            paths.CIRCLE_FEKO_RAW_DIR = original_raw_dir


def test_run_simulation_list_with_real_data():
    """Test FEKO simulation execution with real data validation

    Input: real STL files from tests/data
    Output: validates FEKO execution pipeline
    """
    # Use real test data
    test_data_dir = Path(__file__).parent / "data"
    if not test_data_dir.exists():
        pytest.fail("Test data not available - data directory missing")

    stl_dir = test_data_dir / "circle_stl"
    if not stl_dir.exists():
        pytest.fail("STL test data not available")

    with tempfile.TemporaryDirectory() as temp_dir:
        raw_data_dir = Path(temp_dir) / "raw_data"
        raw_data_dir.mkdir(parents=True, exist_ok=True)

        # Mock the paths
        original_stl_dir = paths.CIRCLE_STL_DIR
        original_raw_dir = paths.CIRCLE_FEKO_RAW_DIR
        paths.CIRCLE_STL_DIR = stl_dir
        paths.CIRCLE_FEKO_RAW_DIR = raw_data_dir

        try:
            # Mock FEKO execution
            with (
                patch(
                    "EMFieldML.FEKO.FEKORunner.FekoRunner.run_solver"
                ) as mock_run_solver,
                patch(
                    "EMFieldML.FEKO.FEKORunner.FekoRunner.delete_warning"
                ) as mock_delete_warning,
            ):

                def mock_run_solver_side_effect(index):
                    # Create dummy FEKO output files when solver is "run"
                    output_file = raw_data_dir / template.raw_data.format(index=index)
                    with open(output_file, "w") as f:
                        f.write("FEKO simulation output\n")
                        f.write("RESULTS FOR LOADS\n")
                        f.write("Power in Watt: 1.0\n")
                        f.write("VALUES OF THE MAGNETIC FIELD STRENGTH in A/m\n")
                        f.write("1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0\n")

                mock_run_solver.side_effect = mock_run_solver_side_effect
                mock_delete_warning.return_value = None

                # Test with real STL files
                test_indices = [1, 41, 364]  # Use existing test data
                FekoRunner.run_simulation_list(test_indices)

                # Verify FEKO was called for each index
                assert mock_run_solver.call_count == len(test_indices)

                # Check that output files were created
                for i in test_indices:
                    output_file = raw_data_dir / template.raw_data.format(index=i)
                    assert output_file.exists(), f"FEKO output file {i} was not created"

        finally:
            paths.CIRCLE_STL_DIR = original_stl_dir
            paths.CIRCLE_FEKO_RAW_DIR = original_raw_dir
