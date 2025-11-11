#Activate for pipeline test, only
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config.configuration import ConfigurationManager
from src.components.model_training import ModelTraining

class ModelTrainingPipeline:
    def __init__(self):
        pass
    def initiate_training(self):
        config=ConfigurationManager()
        model_training_config=config.get_model_training_config()
        model_training=ModelTraining(model_training_config)
        model_training.train()

if __name__=="__main__":
    model_training_pipeline=ModelTrainingPipeline()
    model_training_pipeline.initiate_training()




