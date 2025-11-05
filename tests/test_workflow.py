import os
import pytest
from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from datasets import load_from_disk

@pytest.fixture(scope="session")
def config_manager():
    return ConfigurationManager()

@pytest.mark.skipif(
    os.getenv("CI") == "true", 
    reason="Skipping heavy data download in CI"
)
def test_data_ingestion_creates_local_dataset(config_manager):
    """Medium-heavy test: skip in CI but run locally."""
    cfg = config_manager.get_data_ingestion_config()
    ingestion = DataIngestion(cfg)
    ingestion.download_data()
    assert os.path.exists(cfg.local_data_file), "Dataset file should exist after download."

def test_data_transformation_runs_without_error(config_manager):
    """Lightweight test — always runs."""
    cfg = config_manager.get_data_transformation_config()
    transform = DataTransformation(cfg)
    result = transform.data_transform()
    assert isinstance(result, dict), "Data transformation should return a dictionary."

def test_artifact_structure_exists(config_manager):
    """Sanity test — always runs."""
    ingestion_cfg = config_manager.get_data_ingestion_config()
    assert os.path.exists(ingestion_cfg.root_dir), "Artifact root directory must exist."
