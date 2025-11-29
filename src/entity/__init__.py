from dataclasses import dataclass
from pathlib import Path
@dataclass
class DataIngestionConfig:
    root_dir:Path
    source_url:str
    local_data_file:Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: str

@dataclass
class ModelTrainingConfig:
  root_dir: Path
  data_dir: Path
  model_name: str
  training_params: dict
@dataclass
class ModelEvaluationConfig:
  root_dir: Path
  model_dir: Path
  tokenizer_dir: Path
  data_path: Path
  metric_file_name: str
  




    