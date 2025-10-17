import tempfile
from pathlib import Path

from EMFieldML.config import paths, template
from EMFieldML.Utils.DataSelect import DataSelect


def test_x_train_init():
    """Test initial training data selection

    Input: config parameters (n_shield_shape, selection rules)
    Output: x_train_init.txt (list of selected training data indices)
    """
    assert_file_path = paths.CI_TEST_DATA_DIR / "train_init" / template.x_train_init

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / template.x_train_init
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a temporary directory for the init file too
        init_output_path = Path(temp_dir) / "x_train_init.txt"
        init_output_path.parent.mkdir(parents=True, exist_ok=True)

        DataSelect.generate_x_train_init(
            output_path_init=init_output_path,
            output_path=output_path,
        )
        assert output_path.exists(), "Output file was not created"

        with open(output_path) as f:
            output_data = f.readlines()

        with open(assert_file_path) as f:
            assert_data = f.readlines()

        assert output_data == assert_data, "Content mismatch"


def test_x_test_init():
    """Test initial testing data selection

    Input: config parameters (n_shield_shape, selection rules)
    Output: x_test_init.txt (list of selected testing data indices)
    """
    input_path = paths.CI_TEST_DATA_DIR / "train_init" / template.x_train_init
    assert_file_path = paths.CI_TEST_DATA_DIR / "train_init" / template.x_test_init

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / template.x_test_init
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a temporary directory for the init file too
        init_output_path = Path(temp_dir) / template.x_test_init
        init_output_path.parent.mkdir(parents=True, exist_ok=True)

        DataSelect.generate_x_test_init(
            input_path=input_path,
            output_path_init=init_output_path,
            output_path=output_path,
        )
        assert output_path.exists(), "Output file was not created"

        with open(output_path) as f:
            output_data = f.readlines()

        with open(assert_file_path) as f:
            assert_data = f.readlines()

        assert output_data == assert_data, "Content mismatch"
