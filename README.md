# End-to-End RAG-Based Q&A System for Amazon Automotive Reviews

An MLOps pipeline that ingests raw Amazon automotive review data, preprocesses & embeds it, serves a Retrieval-Augmented Generation (RAG) API + dashboard, and provides hooks for automated evaluation, CI/CD deployment, and monitoring.

---

## Architecture Overview

The system is organized into **five** logical layers—each mapped to a folder or file in this repo:

| Layer                      | Components in This Repo                                      |
|----------------------------|---------------------------------------------------------------|
| **1. Data Ingestion**      | `dags/` (Airflow), `ingest_metadata.py`, `ingest_reviews.py`,<br>`dataset_processor.py` / `process_dataset_cli.py` |
| **2. Model Building**      | `app.py` (Flask API), `dashboard.py` (Flask UI),<br>`static/` + `templates/` |
| **3. Model Evaluation**    | (Future) Log-extraction & RAG-as-a-judge scripts & alerts*    |
| **4. CI/CD & Deployment**  | `Dockerfile.chatbot`, `docker-compose.yml`                   |
| **5. Monitoring & Logging**| Langfuse integration in `webserver_config.py`,<br>GCP Monitoring hooks |

> *_Note: The evaluation layer is scaffolded via Langfuse logs; you can plug in RAG-as-a-judge jobs in `scripts/` as needed._

---

## 1. Data Ingestion

Orchestrated by **Airflow DAGs** in `dags/` (requires `airflow.cfg`, `airflow.db`).

1. **Download / Unzip**  
   - Ingest raw CSV / JSON from S3 or local  
   - See `ingest_metadata.py`, `ingest_reviews.py`  

2. **Schema Validation & Cleaning**  
   - `dataset_processor.validate_and_clean()`  

3. **Data Transformation**  
   - Normalize fields, merge reviews + metadata  

4. **Vector Embedding Generation**  
   - MPNet via HuggingFace (`all-MiniLM-L6-v2`) in `dataset_processor.py`  
   - Outputs embeddings to local Chroma store (or GCS when `CHROMA_DB_DIR` points to a GCS bucket)  

5. **(Optional) Data Bias Report**  
   - Hook in your own report generator in the DAG  

---

## 2. Model Building

Exposed as a **Flask** microservice + dashboard:

- **`app.py`**  
  - `/chat` endpoint for RAG Q&A  
  - **Query Flow**:  
    1. UI (`dashboard.py`) → User query  
    2. **Moderation** via OpenAI →  
    3. **RAG Retriever** (Chroma) →  
    4. **LLM** (OpenAI) →  
    5. **Langfuse** logs the request/response →  
    6. JSON response → UI  

- **`dashboard.py`**  
  - Interactive UI for questions & visualizing results  
  - Templates in `templates/`, static assets in `static/`

- **Configuration**  
  - All API keys, GCS paths, Langfuse DSN → `webserver_config.py` / `.env`

---

## 3. Model Evaluation

While real-time evaluation is plumbed through **Langfuse** (logs every RAG call), you can:

1. **Extract logs** (Langfuse SDK in `scripts/`)  
2. **Run RAG-as-a-judge** (LLM-based metrics)  
3. **Compare** against thresholds  
4. **Trigger** email alerts when performance dips  

> Starter scripts: `scripts/extract_logs.py`, `scripts/evaluate_rag_as_judge.py`

---

## 4. CI/CD & Deployment

- **Local / Dev**  
  ```bash
  docker-compose up --build -d
  ```
  - **Airflow** + **Flask Chatbot** spin up via `docker-compose.yml`  
  - Chatbot image built by `Dockerfile.chatbot`

- **Production Blueprint**  
  1. **GitHub →** push to `main`  
  2. **Build & Push** container to Artifact Registry  
  3. **Deploy** on GKE / Compute Engine  
  4. **On PR**: spin up ephemeral data-pipeline node, auto-destroy on close  

> Future GitHub Actions workﬂows can live in `.github/workflows/`.

---

## 5. Monitoring & Logging

- **Langfuse**  
  - Tracks every API request/response  
  - View error rates, latencies, custom tags  

- **GCP Cloud Monitoring**  
  - Resource utilization dashboards (CPU, Memory)  
  - Alerting rules for above-threshold metrics → Email notifications  

---

## Project Structure

```
.
├── dags/                       # Airflow DAG definitions
├── scripts/                    # Utilities: log extraction, evaluation, helpers
├── static/ & templates/        # Flask dashboard UI
├── Dockerfile.chatbot          # Builds RAG chatbot image
├── docker-compose.yml          # Orchestrates Airflow + Chatbot
├── app.py                      # Flask RAG API & Langfuse integration
├── dashboard.py                # Flask UI server
├── dataset_processor.py        # Validation, cleaning, merging & embedding logic
├── ingest_metadata.py          # Metadata ingestion
├── ingest_reviews.py           # Review ingestion
├── process_dataset_cli.py      # CLI wrapper around dataset_processor
├── webserver_config.py         # Config (.env loader, OpenAI, Langfuse, GCS)
├── requirements.txt            # Python deps
├── airflow.cfg & airflow.db    # Airflow config & metadata DB
└── .env.example                # Copy to `.env` and fill in secrets
```

---

## Getting Started

1. **Clone & configure**  
   ```bash
   git clone <repo-url>
   cd <repo>
   cp .env.example .env
   # Fill in OPENAI_API_KEY, LANGFUSE_DSN, CHROMA_DB_DIR, etc.
   ```

2. **Install / Dockerize**  
   - **Local Python**  
     ```bash
     pip install -r requirements.txt
     airflow db init
     ```
   - **Docker Compose**  
     ```bash
     docker-compose up --build -d
     ```

3. **Airflow UI** → `http://localhost:8080`  
   - Trigger DAGs: metadata → reviews → processing → embeddings  

4. **Dashboard & API** →  
   - API: `http://localhost:5000`  
   - UI:  `http://localhost:5001`

5. **Monitor & Evaluate**  
   - View Langfuse dashboard  
   - Run evaluation scripts in `scripts/`

---

## 🤝 Contributing

1. Fork & branch (`feat/...`, `fix/...`)  
2. Commit with descriptive messages  
3. Open a PR → Review → Merge  
