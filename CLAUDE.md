# Urgent Care Demand Forecast — CLAUDE.md

## Project Overview

End-to-end predictive analytics system forecasting daily patient volume for a fictional Melbourne walk-in clinic ("Southbank Walk-in Clinic"). Built as a data science demonstration covering staffing, triage, supply planning, and wait-time management.

- **Target**: 60–180 patients/day typical; peaks during flu season and post-public-holidays; rare spike days (thunderstorm asthma) up to 400
- **Location**: Melbourne (Southern Hemisphere seasons, Victorian public/school holidays)
- **Data**: 3 years synthetic data (2023–2025), 1,096 daily records

## Architecture

```
urgent-care-demand-forecast/
├── data/
│   ├── raw/clinic_patients_melbourne.csv
│   └── processed/
├── notebooks/                                 # 01_data_simulation → 05_evaluation
├── ml-pipeline/
│   ├── src/data/generate_data.py             # Data generation
│   └── models/                               # Saved artifacts
├── backend/                                  # FastAPI
│   └── app/
│       ├── main.py
│       ├── models.py
│       ├── routes/
│       └── services/
├── frontend/                                 # React + Vite
└── supabase/migrations/
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML/Data | Python 3.10+, Pandas, NumPy, Scikit-learn, XGBoost, Prophet, SHAP |
| Backend | FastAPI, Pydantic v2, Uvicorn |
| Frontend | React 18, Vite, Tailwind CSS, Recharts, Axios |
| Database | Supabase (PostgreSQL) |

## Domain Drivers (vs. typical leisure-venue models)

- **Public holidays INCREASE demand** (GPs closed, walk-in clinics absorb overflow); day-after also spikes
- **Winter drives peaks**, not troughs — flu season Jun–Aug, peak Jul
- **Sunny weather reduces demand**; rainy/cold increases it (respiratory)
- **Temperature extremes both push up** — heat stroke (>35°C) and respiratory (<5°C)
- **Hay fever season** Oct–Nov, amplified by `pollen_index`
- **Thunderstorm asthma** — rare, high-impact shock event (Melbourne-specific, Nov 2016 historical precedent)

## Staffing Tiers

- **Low** (<80 patients): 2 doctors, 3 nurses — standard roster
- **Medium** (80–150): 3 doctors, 4 nurses, extra triage
- **High** (>150): surge protocol — 4+ doctors, overflow coordination with nearest ED

## Development Commands

### ML Pipeline
```bash
cd ml-pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/data/generate_data.py
jupyter notebook
```

### Backend (once built)
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend (once built)
```bash
cd frontend
npm install
npm run dev
```

## Notebook Order

1. `01_data_simulation.ipynb` — generate synthetic clinic data
2. `02_eda.ipynb` — exploratory analysis
3. `03_feature_engineering.ipynb` — cyclical encoding, lags, rolling stats, interactions
4. `04_modeling.ipynb` — Random Forest, XGBoost, Prophet + baselines
5. `05_evaluation.ipynb` — residuals, SHAP, prediction intervals

## ML Evaluation Targets

- MAPE < 15% on test set
- Metrics tracked: MAE, RMSE, R², MAPE, bias

## Behavioral Guidelines

### 1. Think Before Coding
State assumptions. Surface tradeoffs. Ask when uncertain.

### 2. Simplicity First
Minimum code that solves the problem. No speculative features.

### 3. Surgical Changes
Touch only what's needed. Don't "improve" adjacent code.

### 4. Goal-Driven Execution
Define success criteria. Loop until verified.
