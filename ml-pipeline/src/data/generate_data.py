"""
Urgent Care Patient Volume — Synthetic Data Generator
Melbourne walk-in clinic (fictional: "Southbank Walk-in Clinic").

Generates daily patient counts driven by seasonality (flu winter peak,
hay fever spring), weather extremes, public/school holidays (opposite-sign
vs. leisure venues — GPs close, walk-ins spike), and rare shock events
(thunderstorm asthma).
"""

import os

import numpy as np
import pandas as pd


def generate_temperature(month: int) -> float:
    """Melbourne daily mean temperature (degC), by calendar month."""
    temp_map = {
        1: (15, 26, 5),   2: (15, 26, 5),   3: (13, 24, 4),
        4: (11, 20, 4),   5: (8, 16, 3),    6: (6, 14, 3),
        7: (6, 13, 3),    8: (7, 15, 3),    9: (8, 17, 3),
        10: (10, 20, 4), 11: (12, 22, 4),  12: (14, 25, 5),
    }
    min_temp, max_temp, std = temp_map[month]
    avg_temp = (min_temp + max_temp) / 2
    return np.random.normal(avg_temp, std)


def get_weather_type(precip: float) -> str:
    if precip < 0.5:
        return np.random.choice(['Sunny', 'Partly Cloudy'], p=[0.7, 0.3])
    if precip < 5:
        return 'Cloudy'
    return 'Rainy'


def generate_pollen_index(month: int) -> int:
    """Pollen index 0-10. Melbourne grass-pollen peak Oct-Dec."""
    base = {
        1: 2, 2: 2, 3: 1, 4: 1, 5: 0, 6: 0,
        7: 0, 8: 1, 9: 3, 10: 7, 11: 9, 12: 6,
    }[month]
    return int(np.clip(np.random.normal(base, 1.5), 0, 10))


VIC_PUBLIC_HOLIDAYS = [
    # 2023
    '2023-01-01', '2023-01-02', '2023-01-26', '2023-03-13', '2023-04-07',
    '2023-04-08', '2023-04-09', '2023-04-10', '2023-04-25', '2023-06-12',
    '2023-11-07', '2023-12-25', '2023-12-26',
    # 2024
    '2024-01-01', '2024-01-26', '2024-03-11', '2024-03-29', '2024-03-30',
    '2024-03-31', '2024-04-01', '2024-04-25', '2024-06-10', '2024-11-05',
    '2024-12-25', '2024-12-26',
    # 2025
    '2025-01-01', '2025-01-27', '2025-03-10', '2025-04-18', '2025-04-19',
    '2025-04-20', '2025-04-21', '2025-04-25', '2025-06-09', '2025-11-04',
    '2025-12-25', '2025-12-26',
]


def vic_school_holidays() -> list:
    holidays = []
    for year in [2023, 2024, 2025]:
        holidays.extend(pd.date_range(f'{year}-04-01', f'{year}-04-18'))
        holidays.extend(pd.date_range(f'{year}-06-24', f'{year}-07-10'))
        holidays.extend(pd.date_range(f'{year}-09-16', f'{year}-10-02'))
        holidays.extend(pd.date_range(f'{year}-12-18', f'{year}-12-31'))
    return holidays


def generate_clinic_data(
    start_date: str = '2023-01-01',
    end_date: str = '2025-12-31',
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily patient-volume data for a Melbourne walk-in clinic."""
    np.random.seed(seed)
    print(f"Generating data from {start_date} to {end_date}...")

    # --- Calendar scaffold ---
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    df = pd.DataFrame({'date': dates})
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_name'] = df['date'].dt.day_name()
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter

    # --- Calendar drivers ---
    df['is_public_holiday'] = df['date'].astype(str).isin(VIC_PUBLIC_HOLIDAYS).astype(int)
    df['is_day_after_public_holiday'] = df['is_public_holiday'].shift(1).fillna(0).astype(int)
    df['is_school_holiday'] = df['date'].isin(vic_school_holidays()).astype(int)

    # --- Weather ---
    df['temperature'] = df['month'].apply(generate_temperature).round(1)
    df['precipitation'] = np.random.exponential(2, len(df)).round(1)
    df['weather_type'] = df['precipitation'].apply(get_weather_type)
    df['humidity'] = np.clip(np.random.normal(65, 15, len(df)), 20, 100).round(0).astype(int)
    df['pollen_index'] = df['month'].apply(generate_pollen_index)

    df['temp_extreme_hot'] = (df['temperature'] > 35).astype(int)
    df['temp_extreme_cold'] = (df['temperature'] < 5).astype(int)

    # --- Epidemiological drivers (synthetic) ---
    df['is_flu_season'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_flu_peak'] = ((df['month'] == 7) | ((df['month'] == 8) & (df['day'] <= 15))).astype(int)
    df['is_hayfever_season'] = df['month'].isin([10, 11]).astype(int)

    # Thunderstorm asthma — extremely rare high-pollen spring storm shock
    ts_candidates = df[
        (df['pollen_index'] >= 7)
        & (df['weather_type'] == 'Rainy')
        & (df['month'].isin([10, 11, 12]))
    ].index
    df['is_thunderstorm_asthma'] = 0
    n_events = min(len(ts_candidates), int(np.random.poisson(2)))
    if n_events > 0:
        chosen = np.random.choice(ts_candidates, size=n_events, replace=False)
        df.loc[chosen, 'is_thunderstorm_asthma'] = 1

    df['illness_driver_count'] = (
        df['is_flu_season']
        + df['is_hayfever_season']
        + df['is_day_after_public_holiday']
        + df['temp_extreme_hot']
        + df['temp_extreme_cold']
    )

    # --- Patient-volume generation ---
    base = 80
    df['patients'] = float(base)

    dow_multiplier = {0: 1.30, 1: 1.10, 2: 1.00, 3: 1.00, 4: 0.95, 5: 1.15, 6: 1.20}
    df['patients'] *= df['day_of_week'].map(dow_multiplier)

    season_multiplier = {
        1: 0.95, 2: 0.95, 3: 1.00, 4: 1.10, 5: 1.20, 6: 1.40,
        7: 1.50, 8: 1.40, 9: 1.25, 10: 1.15, 11: 1.10, 12: 1.00,
    }
    df['patients'] *= df['month'].map(season_multiplier)

    df.loc[df['is_public_holiday'] == 1, 'patients'] *= 1.40
    df.loc[df['is_day_after_public_holiday'] == 1, 'patients'] *= 1.25
    df.loc[df['is_school_holiday'] == 1, 'patients'] *= 1.10

    df['weather_factor'] = 1.0
    df.loc[df['weather_type'] == 'Sunny', 'weather_factor'] = 0.95
    df.loc[df['weather_type'] == 'Partly Cloudy', 'weather_factor'] = 1.00
    df.loc[df['weather_type'] == 'Cloudy', 'weather_factor'] = 1.05
    df.loc[df['weather_type'] == 'Rainy', 'weather_factor'] = 1.10
    df['patients'] *= df['weather_factor']

    df.loc[df['temp_extreme_hot'] == 1, 'patients'] *= 1.20
    df.loc[df['temp_extreme_cold'] == 1, 'patients'] *= 1.15

    df.loc[df['is_flu_peak'] == 1, 'patients'] *= 1.30
    df.loc[df['is_hayfever_season'] == 1, 'patients'] *= 1.10
    df['patients'] *= 1 + 0.015 * df['pollen_index']

    df.loc[df['is_thunderstorm_asthma'] == 1, 'patients'] *= 2.5

    df['patients'] *= np.random.normal(1.0, 0.12, len(df))
    outlier_idx = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
    df.loc[outlier_idx, 'patients'] *= np.random.uniform(1.4, 2.0, len(outlier_idx))

    days_since_start = (df['date'] - df['date'].min()).dt.days
    df['patients'] *= 1 + (days_since_start * 0.0001)

    df['patients'] = df['patients'].round(0).astype(int).clip(lower=30, upper=500)

    df['patients_last_week'] = df['patients'].shift(7)
    df['patients_last_2weeks'] = df['patients'].shift(14)
    df['patients_last_month'] = df['patients'].shift(30)
    df['patients_7day_avg'] = df['patients'].rolling(window=7, min_periods=1).mean()
    df['patients_14day_avg'] = df['patients'].rolling(window=14, min_periods=1).mean()
    df['patients_30day_avg'] = df['patients'].rolling(window=30, min_periods=1).mean()
    df['patients_same_day_last_year'] = df['patients'].shift(365)

    for col in df.columns:
        if df[col].isna().any() and df[col].dtype in ('float64', 'int64'):
            df[col] = df[col].fillna(df[col].mean())

    final_columns = [
        'date', 'year', 'month', 'day', 'day_of_week', 'day_name', 'week_of_year', 'quarter',
        'is_public_holiday', 'is_day_after_public_holiday', 'is_school_holiday',
        'temperature', 'precipitation', 'weather_type', 'humidity', 'pollen_index',
        'temp_extreme_hot', 'temp_extreme_cold',
        'is_flu_season', 'is_flu_peak', 'is_hayfever_season', 'is_thunderstorm_asthma',
        'illness_driver_count',
        'patients_last_week', 'patients_last_2weeks', 'patients_last_month',
        'patients_7day_avg', 'patients_14day_avg', 'patients_30day_avg',
        'patients_same_day_last_year',
        'patients',
    ]
    df_final = df[final_columns].copy()

    print("\nDataset generated successfully.")
    print(f"Shape: {df_final.shape}")
    p = df_final['patients']
    print(
        f"Patients — mean: {p.mean():.0f}  median: {p.median():.0f}  "
        f"min: {p.min()}  max: {p.max()}"
    )
    print(f"Thunderstorm-asthma events: {int(df_final['is_thunderstorm_asthma'].sum())}")
    return df_final


def main() -> None:
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'raw')
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df = generate_clinic_data()
    output_path = os.path.join(output_dir, 'clinic_patients_melbourne.csv')
    df.to_csv(output_path, index=False)

    print(f"\nSaved to: {output_path}")
    print(f"Total records: {len(df)}")
    print("\nFirst rows:")
    print(df.head())


if __name__ == '__main__':
    main()
