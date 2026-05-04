# Urgent Care Demand Forecast

Daily patient-volume forecasting for a Melbourne walk-in clinic. Synthetic data, open methodology, end-to-end pipeline (data → model → API → dashboard).

## Quickstart

```bash
cd ml-pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/data/generate_data.py
```

Output: `data/raw/clinic_patients_melbourne.csv` (1,096 rows, 2023–2025).

