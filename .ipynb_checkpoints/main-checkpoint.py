from src.logging import logger
from src.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.pipeline.data_transformation_pipeline import DataTransformationPipeline
from src.pipeline.model_training_pipeline import ModelTrainingPipeline


STAGE_NAME="Data Ingestion stage"

try:
    logger.info(f"stage {STAGE_NAME} initiated")
    data_ingestion_pipeline=DataIngestionPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"Stage {STAGE_NAME} Completed")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="Data Transformation stage"

try:
    logger.info(f"stage {STAGE_NAME} initiated")
    data_ingestion_pipeline=DataTransformationPipeline()
    data_ingestion_pipeline.initiate_data_transformation()
    logger.info(f"Stage {STAGE_NAME} Completed")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME="Model training stage"

try:
    logger.info(f"stage {STAGE_NAME} initiated")
    model_training_pipeline=ModelTrainingPipeline()
    model_training_pipeline.initiate_training()
    logger.info(f"Stage {STAGE_NAME} Completed")
except Exception as e:
    logger.exception(e)
    raise e