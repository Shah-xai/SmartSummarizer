from src.entity import ModelEvaluationConfig
from tqdm import tqdm
import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
import pandas as pd
import boto3
import os

class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
        self.config=config
    def generate_batch_size_chunks(self,list_of_elements,batch_size):
        for k in range(0,len(list_of_elements),batch_size):
            yield list_of_elements[k : k+batch_size]

    def get_model_from_s3(self,model_uri, local_dir):
        bucket,prefix=model_uri.replace("s3://", "").split("/", 1)
        s3=boto3.client("s3")
        os.makedirs(local_dir,exist_ok=True)
        paginator=s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket,Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel_path = key[len(prefix):].lstrip("/")
                local_path=os.path.join(local_dir,rel_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                print(f"Downloading {key} -> {local_path}")
                s3.download_file(bucket, key, local_path)
                print(f"\n Download complete → {local_dir}")



    def calculate_metric(self,dataset,
        model,
        tokenizer,
        metric,
        batch_size,
        device="cuda",
        article_column="input",
        summary_column="output"):
        article_batches = list(
            self.generate_batch_size_chunks(dataset[article_column],batch_size))
        summary_batches=list(
            self.generate_batch_size_chunks(dataset[summary_column],batch_size)
        )
        model.eval()
        for article_batch, summary_batch in tqdm(zip(article_batches,summary_batches),total=len(article_batches)):
            input_=tokenizer(article_batch,
                             max_length=1024, truncation=True,
                             padding='max_length',return_tensors="pt"
                             )
            with torch.no_grad():
                summaries=model.generate(input_ids=input_["input_ids"].to(device), 
                                   attention_mask=input_["attention_mask"].to(device),
                                   num_beams=8,
                                   max_length=124,
                                   length_penalty=0.8

                                   )
            
            decoded_summaries=[tokenizer.decode(summary,skip_special_tokens=True,
                                clean_up_tokenization_spaces=True )
                                for summary in summaries]
            decoded_summaries=[d.replace("\n"," ") for d in decoded_summaries]
            metric.add_batch(predictions=decoded_summaries, references=summary_batch)
        return metric.compute()
    def evaluation(self):
        local_dir=os.path.dirname(self.config.model_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_model_from_s3(self.config.root_dir,local_dir)
        tokenizer=AutoTokenizer.from_pretrained(self.config.tokenizer_dir)
        model=AutoModelForSeq2SeqLM.from_pretrained(self.config.model_dir).to(device)
        dataset=load_from_disk(self.config.data_path)
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

        rouge_metric = evaluate.load('rouge')
        score = self.calculate_metric(dataset,model,tokenizer,rouge_metric,batch_size=4,device=device)

        rouge_dict = {rn: score[rn] for rn in rouge_names}

        df = pd.DataFrame(rouge_dict, index = ['pegasus'] )
        df.to_csv(self.config.metric_file_name, index=False)













