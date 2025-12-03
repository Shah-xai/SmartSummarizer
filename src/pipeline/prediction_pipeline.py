from src.config.configuration import ConfigurationManager
from transformers import pipeline, AutoTokenizer
import torch


class PredictionPipeline:
    def __init__(self):
        config = ConfigurationManager().get_model_evaluation_config()

        # Load tokenizer and model once
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_dir)

        device = 0 if torch.cuda.is_available() else -1

        self.pipe = pipeline(
            task="summarization",
            model=config.model_dir,      # local dir with pegasus-model
            tokenizer=self.tokenizer,
            device=device,
        )

        # Default generation params
        self.gen_kwargs = {
            "length_penalty": 0.85,
            "num_beams": 4,
            "max_length": 128,
           
        }

    def predict(self, text: str) -> str:
        print("Original text:")
        print(text)

        summary = self.pipe(text, **self.gen_kwargs)[0]["summary_text"]

        print("\nModel Summary:")
        print(summary)

        return summary
