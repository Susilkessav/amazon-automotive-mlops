# Amazon Automotive MLOps

A containerized end‑to‑end MLOps project for Amazon automotive reviews, featuring:

- **Data ingestion & preprocessing** (HuggingFace `datasets`, pandas)  
- **Model training & evaluation** (scikit‑learn, RandomForest)  
- **Embedding & chatbot pipelines** (Sentence‑Transformers, FAISS, OpenAI)  
- **Streamlit**‑powered **ML API** and **Chatbot UI**  
- **Airflow** orchestration of all pipelines  
- **Docker Compose** for easy local/dev setup  

---

## Table of Contents

1. [Features](#features)  
2. [Repository Structure](#repository-structure)  
3. [Prerequisites](#prerequisites)  
4. [Local Development](#local-development)  
5. [Docker‑Compose Full Stack](#docker-compose-full-stack)  
6. [Workflow Overview](#workflow-overview)  
7. [Contributing](#contributing)  
8. [License](#license)  

---

## Features

1. **Data Pipeline**  
   - Ingest raw reviews & metadata  
   - Join, clean & produce train/val/test splits  

2. **Model Pipeline**  
   - TF‑IDF + RandomForest regression  
   - Train/test evaluation, artifact save  

3. **Embedding + Chatbot Pipeline**  
   - Precompute sentence embeddings (all‑MiniLM‑L6‑v2)  
   - Build FAISS index for retrieval  
   - Flask Chat API with context‑grounded answers  

4. **ML API** (Streamlit)  
   - Predictive app for rating regression on port 8501  

5. **Chatbot UI** (Streamlit)  
   - Conversational frontend on port 8502  
   - Backend API on port 5001  

6. **Airflow Orchestration**  
   - DAGs for data, model, deploy & chatbot pipelines  
   - Fully automated end‑to‑end flow  

7. **Dockerized**  
   - Single command bring‑up of the full stack  

---

## Repository Structure

```
.
├── dags/                     # Airflow DAG definitions
│   ├── data_pipeline.py
│   ├── model_pipeline.py
│   └── chatbot_pipeline.py
├── data/                     # Data mounts (raw, processed)
│   ├── reviews/
│   └── metadata/
├── models/                   # Saved models & indices
│   ├── model.joblib
│   └── chatbot/
│       ├── faiss.index
│       └── contexts.pkl
├── scripts/                  # ETL, training, embedding, ingestion
│   ├── ingest_reviews.py
│   ├── ingest_metadata.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── compute_embeddings.py
│   └── chatbot_service.py
├── app/                      # Streamlit frontends
│   ├── streamlit_app.py     # ML API
│   └── streamlit_chat.py    # Chat UI
├── Dockerfile.train          # training container
├── Dockerfile.inference      # ML API container
├── Dockerfile.chatbot        # chatbot container
├── docker-compose.yml        # full‑stack Compose
├── requirements.txt          # Python deps for training & API
├── requirements_airflow.txt  # Python deps for Airflow DAGs
├── .gitignore
└── README.md
```

---

## Prerequisites

- Docker & Docker Compose  
- (Optional) Python 3.10+ & virtualenv for local testing  
- An OpenAI API key (for the chatbot)

---

## Local Development

1. **Clone & .gitignore**  
   ```bash
   git clone https://github.com/<YOUR‑USER>/amazon-automotive-mlops.git
   cd amazon-automotive-mlops
   ```

2. **(Optional) Python venv**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run ingestion & training**  
   ```bash
   python scripts/ingest_reviews.py
   python scripts/ingest_metadata.py
   python scripts/preprocess.py
   python scripts/train_model.py
   python scripts/compute_embeddings.py
   ```

4. **Test chatbot locally**  
   ```bash
   export OPENAI_API_KEY="sk-..."
   python scripts/chatbot_service.py
   curl -X POST http://127.0.0.1:5001/chat \
     -H "Content-Type: application/json" \
     -d '{"query":"Which automotive battery has the highest rating?"}'
   ```

---

## Docker‑Compose Full Stack

Bring up **Airflow**, **ML API**, and **Chatbot** together:

1. **Set your OpenAI key**  
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

2. **Tear down & rebuild**  
   ```bash
   docker-compose down
   docker-compose up --build -d
   ```

3. **Verify services**  
   - **Airflow UI** → http://localhost:8080  
     - Login: `admin` / `admin`  
     - No “scheduler” banner  
     - Trigger DAGs from the UI  
   - **ML API** → http://localhost:8501  
   - **Chatbot UI** → http://localhost:8502  
   - **Chat API** → http://localhost:5001/chat  

---

## Workflow Overview

1. **Airflow DAGs**  
   - **data_pipeline**: ingest → preprocess  
   - **model_pipeline**: train → evaluate  
   - **deploy_pipeline**: package → serve  
   - **chatbot_pipeline**: embed → index refresh  

2. **Streamlit ML API** serves regression predictions.  
3. **Chatbot** uses FAISS + OpenAI for Q&A.  

Re‑trigger any DAG in Airflow to re‑run that pipeline stage.

---
