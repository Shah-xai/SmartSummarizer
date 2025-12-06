# SmartSummarizer — End-to-End Text Summarization (FastAPI + HuggingFace + AWS SageMaker)

SmartSummarizer is a full-stack summarization project demonstrating a production-style pipeline:  
**data ingestion → preprocessing → model training → evaluation → deployment → API-based inference.**  
It’s meant as a **showcase / demo project** — the core code, configs, and API logic are included, but **pre-trained model artifacts are omitted**.  

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
├──SmartSummarizer/
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

##  Evaluation

- **ROUGE-1**
- **ROUGE-2**
- **ROUGE-L**

##  Demo Mode & Model Artifacts

- This repository does **not** include the large model weights (≈ 5 GB) to keep the repo lightweight.  
- Instead, a **demo video** illustrates how the system works end-to-end (inference via FastAPI).  

## 🎬 Demo Video

[![Watch SmartSummarizer Demo](video_cover.jpg)](https://youtu.be/D-3VGY7jPXE)  

---

## 👍 Quick Start — Run in Demo Mode (no model weights)

1. Clone the repository
2. Update config.yaml   
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
4. Run the main file
    ```bash
    python main.py

##  Author

**Shahriyar**  
Machine Learning Engineer
