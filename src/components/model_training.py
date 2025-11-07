from src.entity import ModelTrainingConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq,Trainer
from transformers.training_args import TrainingArguments
from datasets import load_from_disk
import torch
import os

class ModelTraining:
    def __init__(self,config:ModelTrainingConfig):
        self.config=config
        
    def train(self):
        device ="cuda" if torch.cuda.is_available() else "cpu"
        tokenizer=AutoTokenizer.from_pretrained(self.config.model_name)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name).to(device)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_pegasus)
        # loading data
        data_set = load_from_disk(self.config.data_dir)


        # Training setup
        trainer_args = TrainingArguments(
                        output_dir=self.config.root_dir,
                        **self.config.training_params)
      

            
        model_trainer=Trainer(model=model_pegasus,args=trainer_args,
                              data_collator=data_collator, 
                              tokenizer=tokenizer,
                              train_dataset=data_set["train"],
                              eval_dataset=data_set["validation"]

        )

        model_trainer.train()
        # Save model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir,"pagasus-model"))
        # Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))







