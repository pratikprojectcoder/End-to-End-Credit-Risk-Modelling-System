# End-to-End Credit Risk Modeling System

## Layout

- `data/raw/` — unmodified source datasets  
- `data/processed/` — cleaned, feature-engineered tables ready for modeling  
- `notebooks/` — `eda.ipynb`, `preprocessing.ipynb`, `modeling.ipynb`  
- `src/` — `preprocess.py`, `feature_engineering.py`, `train_model.py`, `evaluate_model.py`, `explain_model.py`, `predict.py`, `utils.py`  
- `models/` — trained `xgboost_model.pkl`, `scaler.pkl`, `label_encoders.pkl` (created by training or seeded for demos)  
- `app/streamlit_app.py` — simple scoring UI  
- `main.py` — CLI (`train`, `evaluate`, `explain`)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Commands

- Train (expects `data/processed/train.parquet` with a `target` column):  
  `python main.py train`  
  Optional: `python main.py train --data path\to\train.parquet --target my_label`
- Evaluate holdout:  
  `python main.py evaluate`  
  (default `data/processed/test.parquet`)
- Feature importances:  
  `python main.py explain`
- Streamlit:  
  `streamlit run app/streamlit_app.py`

The seeded `models/*.pkl` files use placeholder features `f0`, `f1`, `f2` until you run `main.py train` on your real processed data.

Run Jupyter from the project root to use the notebooks.
