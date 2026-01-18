"""
Export Model Artifacts for Web Application

This script exports the trained model and risk factors so the web app
can use them without re-running the entire training pipeline.

Outputs:
1. route_risk_factors.json - Historical delay metrics per route
2. stop_risk_factors.json - Historical delay metrics per stop
3. model.pkl - The trained logistic regression model
4. scaler.pkl - The feature scaler
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

print("=" * 70)
print("EXPORTING MODEL ARTIFACTS FOR WEB APP")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/5] Loading parsed data...")

csv_files = sorted(Path("data/interim/gtfs_rt").glob("trip_updates_parsed_*.csv"))
if not csv_files:
    raise FileNotFoundError("No parsed CSV files found. Run parse script first.")

data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Add temporal features
data['timestamp'] = pd.to_datetime(data['feed_timestamp'], unit='s')
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
data['is_rush_hour'] = data['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)

print(f"Loaded {len(data):,} records")

# ============================================================================
# SPLIT DATA (to avoid leakage)
# ============================================================================
print("\n[2/5] Splitting data for feature calculation...")

data_sorted = data.sort_values('timestamp').reset_index(drop=True)
split_idx = int(len(data_sorted) * 0.8)

train_df = data_sorted.iloc[:split_idx].copy()
test_df = data_sorted.iloc[split_idx:].copy()

# ============================================================================
# CALCULATE RISK FACTORS (on training data only)
# ============================================================================
print("\n[3/5] Calculating risk factors...")

# Route statistics
route_stats = train_df.groupby('route_id').agg({
    'delay_10plus': ['mean', 'count'],
    'delay_min': ['mean', 'std']
}).reset_index()
route_stats.columns = ['route_id', 'delay_rate', 'trip_count', 'avg_delay', 'std_delay']
route_stats['std_delay'] = route_stats['std_delay'].fillna(0)

# Stop statistics
stop_stats = train_df.groupby('stop_id').agg({
    'delay_10plus': ['mean', 'count'],
    'delay_min': ['mean']
}).reset_index()
stop_stats.columns = ['stop_id', 'delay_rate', 'trip_count', 'avg_delay']

print(f"  Route stats: {len(route_stats)} routes")
print(f"  Stop stats: {len(stop_stats)} stops")

# ============================================================================
# TRAIN MODEL (for export)
# ============================================================================
print("\n[4/5] Training model for export...")

# Merge stats back
global_route_delay_rate = train_df['delay_10plus'].mean()
global_route_avg_delay = train_df['delay_min'].mean()
global_stop_delay_rate = train_df['delay_10plus'].mean()
global_stop_avg_delay = train_df['delay_min'].mean()

def apply_stats(df, r_stats, s_stats):
    df = df.merge(r_stats, on='route_id', how='left')
    df = df.merge(s_stats, on='stop_id', how='left', suffixes=('_route', '_stop'))
    
    df['delay_rate_route'] = df['delay_rate_route'].fillna(global_route_delay_rate)
    df['avg_delay_route'] = df['avg_delay_route'].fillna(global_route_avg_delay)
    df['std_delay'] = df['std_delay'].fillna(0)
    
    df['delay_rate_stop'] = df['delay_rate_stop'].fillna(global_stop_delay_rate)
    df['avg_delay_stop'] = df['avg_delay_stop'].fillna(global_stop_avg_delay)
    
    return df

train_df = apply_stats(train_df, route_stats, stop_stats)
test_df = apply_stats(test_df, route_stats, stop_stats)

# Feature columns
feature_cols = [
    'hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
    'avg_delay_route', 'std_delay', 'delay_rate_route',
    'avg_delay_stop', 'delay_rate_stop', 'stop_sequence'
]

X_train = train_df[feature_cols].copy()
y_train = train_df['delay_10plus'].values

X_test = test_df[feature_cols].copy()
y_test = test_df['delay_10plus'].values

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)
model.fit(X_train_scaled, y_train)

print("  Model trained successfully")

# ============================================================================
# EXPORT ARTIFACTS
# ============================================================================
print("\n[5/5] Exporting artifacts...")

export_dir = Path("web_app/artifacts")
export_dir.mkdir(parents=True, exist_ok=True)

# 1. Route risk factors
route_risk_dict = route_stats.to_dict('records')
with open(export_dir / "route_risk_factors.json", 'w') as f:
    json.dump(route_risk_dict, f, indent=2)
print(f"  ✅ Exported route_risk_factors.json ({len(route_risk_dict)} routes)")

# 2. Stop risk factors
stop_risk_dict = stop_stats.to_dict('records')
with open(export_dir / "stop_risk_factors.json", 'w') as f:
    json.dump(stop_risk_dict, f, indent=2)
print(f"  ✅ Exported stop_risk_factors.json ({len(stop_risk_dict)} stops)")

# 3. Model
joblib.dump(model, export_dir / "model.pkl")
print(f"  ✅ Exported model.pkl")

# 4. Scaler
joblib.dump(scaler, export_dir / "scaler.pkl")
print(f"  ✅ Exported scaler.pkl")

# 5. Feature names (for reference)
with open(export_dir / "feature_names.json", 'w') as f:
    json.dump(feature_cols, f, indent=2)
print(f"  ✅ Exported feature_names.json")

# 6. Global defaults (for unknown routes/stops)
global_defaults = {
    "route_delay_rate": float(global_route_delay_rate),
    "route_avg_delay": float(global_route_avg_delay),
    "stop_delay_rate": float(global_stop_delay_rate),
    "stop_avg_delay": float(global_stop_avg_delay)
}
with open(export_dir / "global_defaults.json", 'w') as f:
    json.dump(global_defaults, f, indent=2)
print(f"  ✅ Exported global_defaults.json")

print("\n" + "=" * 70)
print("EXPORT COMPLETE!")
print("=" * 70)
print(f"\nArtifacts saved to: {export_dir.absolute()}/")
print("\nNext steps:")
print("  1. Build Flask backend using these artifacts")
print("  2. Create prediction endpoint that loads model.pkl")
print("  3. Build React frontend to query the API")
