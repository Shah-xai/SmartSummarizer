from src.entity import DataTransformationConfig
from src.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk
import os

class DataTransformation:
    def __init__(self,config:DataTransformationConfig):
        self.config=config
        self.tokenizer=AutoTokenizer.from_pretrained(self.config.tokenizer_name)
    def data_tokenizer(self,data_batch):
    
        input_encoder=self.tokenizer(
            data_batch["input"],
            max_length=1024,
            truncation=True
        )
        with self.tokenizer.as_target_tokenizer():
            target_encoder = self.tokenizer(data_batch["output"],max_length=128,truncation=True)
        return {
            'input_ids' : input_encoder['input_ids'],
        'attention_mask': input_encoder['attention_mask'],
        'labels': target_encoder['input_ids']
        }   
    def data_transformer(self):
        file_path = self.config.data_path
        logger.info ("Loading data for tokenization...")
        data= load_from_disk(file_path) 
        data_tokenized=data.map(self.data_tokenizer, batched=True)
        logger.info("Data tokenization completed!")
        data_tokenized.save_to_disk(os.path.join(self.config.root_dir,"Samsung-samsum-transformed"))