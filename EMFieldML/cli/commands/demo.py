"""Demo functionality for EMFieldML electromagnetic field toolkit.

This module provides demo functionality to showcase the capabilities
of the electromagnetic field machine learning toolkit.
"""

import sys
import zipfile
from pathlib import Path

import click
import gdown

from EMFieldML.config import paths


def download_data() -> bool:
    """Download demo data from Google Drive."""
    demo_file_id = "1oZ-xCOtIjiFQDlrd1jn7SFNiWHk0x2wu"
    demo_filename = "demo_data.zip"

    try:
        if Path("demo_data").exists():
            return True

        if not Path(demo_filename).exists():
            click.echo("Downloading demo data...")
            gdown.download(
                f"https://drive.google.com/uc?id={demo_file_id}",
                demo_filename,
                quiet=False,
                fuzzy=True,
            )

        click.echo("Extracting demo data...")
        with zipfile.ZipFile(demo_filename, "r") as zip_ref:
            zip_ref.extractall(".")

        Path(demo_filename).unlink()
        return Path("demo_data").exists()

    except Exception as e:
        click.echo(f"Download failed: {e}")
        return False


def patch_config() -> None:
    """Patch configuration to use demo data directory."""
    demo_data_dir = Path("demo_data/data")
    paths.DATA_DIR = demo_data_dir
    paths.CIRCLE_MOVE_X_DIR = demo_data_dir / "circle_move_x_data"
    paths.CIRCLE_STL_DIR = demo_data_dir / "stl"
    paths.Y_DATA_DIR = demo_data_dir / "y_data"
    paths.TRAIN_DIR = demo_data_dir / "train"
    paths.LEARNING_MODEL_DIR = demo_data_dir / "learning_model"
    paths.DEFORMATION_DIR = demo_data_dir / "deformation"


@click.command()
@click.option(
    "--timeout",
    type=int,
    default=0,
    help="Auto-close window after N seconds (0 = stay open)",
)
def demo(timeout: int) -> None:
    """Run EMFieldML demo for shield #1."""
    try:
        if not Path("demo_data").exists() and not download_data():
            sys.exit(1)
        patch_config()
        # Import Visualize after patching config to ensure it uses the patched paths
        from EMFieldML.Visualize import Visualize

        visualize = Visualize(1)
        visualize.visualize(timeout_seconds=timeout)
    except Exception as e:
        click.echo(f"Error: {e}")
        sys.exit(1)
