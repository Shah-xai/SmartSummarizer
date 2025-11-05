from src.entity import DataIngestionConfig
import os
from src.utils.common import read_yaml, create_directories
from src.constants import *

class ConfigurationManager():
    def __init__(self, config_path=CONFIG_FILE_PATH, param_path=PARAM_FILE_PATH):
        self.config=read_yaml(config_path)
        self.param=read_yaml(param_path)
        create_directories([self.config.artifacts_dir])
    def get_data_ingestion_config(self)->DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file
        )
        return data_ingestion_config
    
