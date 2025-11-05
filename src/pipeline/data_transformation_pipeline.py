# Activate for pipeline test, only
# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))




from src.config.configuration import ConfigurationManager
from src.components.data_transformation import DataTransformation

class DataTransformationPipeline:
    def __init__(self):
        pass
    def initiate_data_transformation(self):
        config=ConfigurationManager()
        data_transformation_config=config.get_data_transformation_config()
        data_transformation=DataTransformation(data_transformation_config)
        data_transformation.data_transformer()
    
if __name__=="__main__":
    
    data_transformation_pipeline=DataTransformationPipeline()
    data_transformation_pipeline.initiate_data_transformation()
    



