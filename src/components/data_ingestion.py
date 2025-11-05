from src.entity import DataIngestionConfig
from datasets import load_dataset
from src.logging import logger
import os

class DataIngestion():
    def __init__(self,config:DataIngestionConfig):
        self.config=config
    def download_data(self):
        data_path=self.config.source_url
        if not os.path.exists(self.config.local_data_file):
            logger.info(f" Downloading data into {self.config.local_data_file}...")
            ds = load_dataset(data_path)
            ds.save_to_disk(self.config.local_data_file)
            
        else:
            logger.info("Dataset already exists!")


