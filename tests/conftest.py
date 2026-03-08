"""
Pytest configuration and shared fixtures for FlowFigTabMiner tests.
"""
import pytest
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def fixtures_dir():
    """Return the fixtures directory."""
    return PROJECT_ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def sample_table_image(fixtures_dir):
    """Return path to a sample table image for testing."""
    return fixtures_dir / "sample_table.png"


@pytest.fixture(scope="session")
def sample_figure_image(fixtures_dir):
    """Return path to a sample figure image for testing."""
    return fixtures_dir / "sample_figure.png"


@pytest.fixture(scope="session")
def test_output_dir(tmp_path_factory):
    """Create a temporary output directory for tests."""
    return tmp_path_factory.mktemp("test_output")


@pytest.fixture
def mock_config():
    """Return a mock configuration for testing."""
    return {
        "global": {
            "output_base_dir": "data/output",
            "intermediate_dir": "data/intermediate",
            "device": "cpu"
        },
        "tables": {
            "segmentation": {"model_path": "models/test_table_seg.pt", "threshold": 0.8},
            "molecule_detection": {"model_path": "models/test_mol_det.pt", "confidence_threshold": 0.25},
            "structure": {"model_path": "microsoft/table-transformer-structure-recognition-v1.1-all"},
        },
        "figures": {
            "step2_macro": {"model_path": "models/test_fig_macro.pt", "confidence_threshold": 0.5},
            "step3_micro": {"model_path": "models/test_fig_micro.pt", "confidence_threshold": 0.25},
        }
    }
