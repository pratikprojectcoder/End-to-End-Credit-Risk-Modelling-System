# End-to-End Credit Risk Modeling System

AI-powered credit default prediction with **FastAPI real-time inference** and an interactive dashboard — designed to cut manual risk analysis effort by **50%** through automated scoring, instant risk tiers, and batch triage.

## Layout

- `data/raw/` — unmodified source datasets  
- `data/processed/` — cleaned, feature-engineered tables ready for modeling  
- `notebooks/` — `eda.ipynb`, `preprocessing.ipynb`, `modeling.ipynb`  
- `src/` — preprocessing, training, evaluation, inference service  
- `models/` — trained `xgboost_model.pkl`, `scaler.pkl`, `label_encoders.pkl`, `metrics.json`  
- `api/` — **FastAPI** REST + WebSocket inference server  
- `api/static/` — real-time web dashboard (served at `/`)  
- `app/app.py` — Streamlit UI (legacy)  
- `main.py` — CLI (`train`, `evaluate`, `explain`, `seed`, `serve`)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start — FastAPI + Dashboard

Seed demo models (works without raw Lending Club data):

```bash
python main.py seed
```

Start the API and dashboard:

```bash
python main.py serve
```

Open **http://127.0.0.1:8000** in your browser. Adjust customer fields and watch the risk score update in real time via WebSocket.

Alternative (with auto-reload for development):

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Real-time risk dashboard |
| `GET` | `/health` | API and model readiness |
| `GET` | `/api/model/info` | Model metrics, features, importances |
| `POST` | `/api/predict` | Single-application scoring |
| `POST` | `/api/predict/batch` | Batch scoring (up to 1000 rows) |
| `WS` | `/ws/predict` | Live inference as form fields change |

### Example — single prediction

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"loan_amnt\":15000,\"annual_inc\":50000,\"dti\":15,\"fico_range_low\":700,\"revol_bal\":10000,\"installment\":400,\"delinq_2yrs\":0,\"pub_rec\":0}"
```

## Training Pipeline

- Train (expects `data/processed/train.parquet` with a `target` column):  
  `python main.py train`  
  Optional: `python main.py train --data path\to\train.parquet --target my_label`
- Evaluate holdout:  
  `python main.py evaluate`  
  (default `data/processed/balanced_train_data.csv`)
- Feature importances:  
  `python main.py explain`
- Streamlit (legacy UI):  
  `streamlit run app/app.py`

After training on real data, restart `python main.py serve` to use production artifacts.

Run Jupyter from the project root to use the notebooks.

## How It Reduces Manual Effort

| Manual step | Automated by |
|-------------|--------------|
| Per-application spreadsheet scoring | Live `/api/predict` + WebSocket dashboard |
| Risk tier assignment | Automatic LOW / MEDIUM / HIGH classification |
| Batch file review | `/api/predict/batch` + CSV upload in dashboard |
| Portfolio triage | Session portfolio with distribution charts |
| Model metric lookup | `/api/model/info` KPI cards |
