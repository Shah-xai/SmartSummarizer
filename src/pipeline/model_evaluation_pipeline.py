# #Activate for pipeline test, only
# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import ModelEvaluation
class ModelEvaluationPipeline:
    def __init__(self):
        pass
    def initiate_model_evaluation(self):
        config=ConfigurationManager()
        evaluation_config=config.get_model_evaluation_config()
        model_eval=ModelEvaluation(evaluation_config)
        model_eval.evaluation()



if __name__=="__main__":
    model_eval_pipeline=ModelEvaluationPipeline()
    model_eval_pipeline.initiate_model_evaluation()