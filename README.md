# SmartSummarizer — End-to-End LLM Summarization System (FastAPI + HuggingFace + AWS SageMaker)

SmartSummarizer is a full-stack text-summarization system built with production-grade machine learning engineering principles.
It combines **HuggingFace transformer models**, **AWS SageMaker training**, **S3 model storage**, and a **FastAPI web backend**
to deliver reliable summarization services at scale.

This project demonstrates the complete ML lifecycle:
**data ingestion → preprocessing → model training → evaluation → deployment → API-based inference.**

##  Features

- **Modern LLM Architecture** built on HuggingFace Transformers  
- **FastAPI Backend** with clean REST endpoints (`/summarize`)  
- **SageMaker Training Pipeline** with S3 integration for artifacts  
- **Config-driven Architecture** using YAML for reproducibility  
- **Modular ML Code** following a scalable `src/` layout  
- **Container-ready** for deployment  
- **CI-friendly Repo** with clear dependencies + `.gitignore`

## System Overview

1. User sends text to your **FastAPI endpoint**  
2. API passes input to the summarization model  
3. Model runs **transformer-based abstractive summarization**  
4. API returns a concise, human-readable summary  
5. Optional: Endpoint calls a **SageMaker-hosted model** for cloud inference  

## 📂 Project Structure

```
SmartSummarizer/
├──
│   ├── main.py
│   └── routers/
├── src/
│   ├── data/
│   ├── model/
│   ├── utils/
│   └── config/
├── sagemaker/
├── requirements.txt
├── README.md
└── .gitignore
```

## 🛠 Tech Stack

| Category | Tools |
|---------|--------|
| **Language** | Python 3.10 |
| **ML Framework** | HuggingFace Transformers, PyTorch |
| **API Framework** | FastAPI |
| **Cloud ML** | AWS SageMaker, S3 |
| **Containerization** | Docker |
| **Dev Tools** | GitHub |

##  Training Pipeline (AWS SageMaker)

Includes:

- Tokenization + preprocessing  
- Transformer model fine-tuning  
- Gradient clipping & warmup  
- Early stopping  
- Artifact saving to S3  
- Optional real-time inference endpoint  

##  Evaluation

- **ROUGE-1**
- **ROUGE-2**
- **ROUGE-L**

## 🌐 FastAPI Web Application

Start the API:

```bash
python app.py
```


## Deployment Options

### 1. Local
- Run FastAPI 
- Load model locally  

### 2. Cloud
- Dockerize  
- Deploy to ECS / Fargate / EC2 / Lambda  
- Use SageMaker endpoint for inference  

## Roadmap

- Long-context summarization  
- RAG document summarization  
- Batch summarization API  
- Add frontend UI  
- Add CloudWatch monitoring  

##  Author

**Shahriyar**  
Machine Learning Engineer
