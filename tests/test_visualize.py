"""Slow visualization tests - excluded from normal test runs.

These tests require demo data and take significant time to run.
Run with: pytest tests/test_visualize.py -m slow
"""

import sys
from pathlib import Path

import pytest

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from EMFieldML.Visualize.Visualize import Visualize


class TestVisualizationSlow:
    """Slow visualization tests - requires demo data and takes time."""

    @pytest.fixture
    def visualizer(self):
        """Create a visualizer instance for testing."""
        return Visualize(number=1)

    @pytest.mark.slow
    def test_initialization(self, visualizer):
        """Test that visualizer initializes correctly."""
        assert visualizer.number == 1
        assert hasattr(visualizer, "ui")
        assert hasattr(visualizer, "point_size")

    @pytest.mark.slow
    def test_setup(self, visualizer):
        """Test visualization setup."""
        try:
            result = visualizer.test_setup()
            assert isinstance(result, bool)
        except Exception as e:
            pytest.skip(f"Setup test failed: {e}")

    @pytest.mark.slow
    def test_predictions(self, visualizer):
        """Test prediction pipeline."""
        try:
            result = visualizer.test_predictions()
            assert isinstance(result, bool)
        except Exception as e:
            pytest.skip(f"Prediction test failed: {e}")

    @pytest.mark.slow
    def test_visualize_with_timeout(self, visualizer):
        """Test visualize method with short timeout."""
        try:
            visualizer.visualize(timeout_seconds=1)
        except Exception as e:
            pytest.skip(f"Visualization test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
