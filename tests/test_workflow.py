import os
import pytest
from datasets import load_from_disk
from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

pytest.skip("Skipping heavy data download in CI", allow_module_level=True)

@pytest.fixture(scope="session")
def config_manager():
    """Create a shared ConfigurationManager for all tests."""
    return ConfigurationManager()


def test_data_ingestion_creates_local_dataset(config_manager):
    """Ensure that the dataset gets downloaded and saved properly."""
    ingestion_config = config_manager.get_data_ingestion_config()
    data_ingestion = DataIngestion(ingestion_config)

    # Run the data download (safe if already cached)
    data_ingestion.download_data()

    # Assert the saved dataset exists
    assert os.path.exists(ingestion_config.local_data_file), "Local dataset file not found"

    # Check it’s a valid HF dataset
    ds = load_from_disk(ingestion_config.local_data_file)
    assert "train" in ds and "validation" in ds, "Dataset structure incomplete"


def test_data_transformation_runs_without_error(config_manager):
    """Ensure data transformation loads and tokenizes correctly."""
    transform_config = config_manager.get_data_transformation_config()
    data_transform = DataTransformation(transform_config)

    # Run the transformation method
    try:
        result = data_transform.data_transform()
    except Exception as e:
        pytest.fail(f"Data transformation failed with error: {e}")

    # Check the structure of the output
    assert isinstance(result, dict), "Data transformation did not return a dictionary"
    for key in ["input_ids", "attention_mask", "labels"]:
        assert key in result, f"Missing key '{key}' in transformation output"


def test_artifact_structure_exists(config_manager):
    """Check that artifact directories exist (created during ingestion)."""
    ingestion_config = config_manager.get_data_ingestion_config()
    root_dir = ingestion_config.root_dir
    assert os.path.exists(root_dir), "Artifacts directory missing"
    assert any(os.scandir(root_dir)), "Artifacts directory is empty"
