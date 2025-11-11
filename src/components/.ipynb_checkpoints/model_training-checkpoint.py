from src.entity import ModelTrainingConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
from transformers import DataCollatorForSeq2Seq,Trainer, EarlyStoppingCallback
from transformers.training_args import TrainingArguments
from datasets import load_from_disk
from src.logging import logger
import boto3
import torch
import os

class ModelTraining:
    def __init__(self,config:ModelTrainingConfig):
        self.config=config
        
    def train(self):
        set_seed(42)
        device ="cuda" if torch.cuda.is_available() else "cpu"
        tokenizer=AutoTokenizer.from_pretrained(self.config.model_name)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name).to(device)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_pegasus)
        # loading data
        data_set = load_from_disk(self.config.data_dir)
        # Local temp output dir
        local_output_dir = "/tmp/summarizer_training"
        os.makedirs(local_output_dir, exist_ok=True)

        trainer_args = TrainingArguments(
            output_dir=local_output_dir,
            logging_dir=os.path.join(local_output_dir, "logs"),
            save_strategy="steps",
            save_total_limit=1,
            **self.config.training_params,
        )

        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            data_collator=data_collator,
            tokenizer=tokenizer,
            train_dataset=data_set["train"],
            eval_dataset=data_set["validation"],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )

        trainer.train()
        logger.info("Training complete locally.")

        # Save model locally first
        model_dir = os.path.join(local_output_dir, "pegasus-model")
        tokenizer_dir = os.path.join(local_output_dir, "tokenizer")

        model_pegasus.save_pretrained(model_dir)
        tokenizer.save_pretrained(tokenizer_dir)
        # Upload to S3
        s3_output_dir = self.config.root_dir
        bucket, key_prefix = s3_output_dir.replace("s3://", "").split("/", 1)
        s3 = boto3.client("s3")
        
        def upload_folder(local_folder, remote_prefix):
            for root, _, files in os.walk(local_folder):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, local_folder)
                    s3.upload_file(local_path, bucket, f"{key_prefix}/{remote_prefix}/{relative_path}")
        
        logger.info(" Uploading model artifacts to S3...")
        upload_folder(model_dir, "pegasus-model")
        upload_folder(tokenizer_dir, "tokenizer")
        
        logger.info(f" All artifacts successfully uploaded to {s3_output_dir}")





