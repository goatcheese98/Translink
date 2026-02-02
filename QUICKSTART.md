# Quick Start Guide

## 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r config/requirements.txt

# Copy environment template
cp config/.env.example .env
# Edit .env and add your TransLink API key
```

## 2. Collect Data

```bash
# Collect one sample locally
python -m src.data.collector --local --type trip_updates

# Or process existing data
python -m src.data.parser data/raw/gtfs_rt/trip_updates --output data/processed
```

## 3. Engineer Features

```bash
# Create features from parsed data
python -m src.data.features data/processed --output data/features/features.csv
```

## 4. Train Model

```bash
# Train and save model
python scripts/train_model.py --data-dir data/processed --output-dir artifacts/models

# Or use the module directly
python -m src.models.train data/features/features.csv --output-dir artifacts/models
```

## 5. Run API Locally

```bash
# Set model path
export MODEL_PATH=artifacts/models/model_YYYYMMDD_HHMMSS.pkl

# Start API
python -m src.api.main

# Test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"stop_id": "1234", "route_id": "099"}'
```

## 6. Deploy to GCP (Future)

See ARCHITECTURE.md for GCP deployment steps.

## Project Commands

| Command | Purpose |
|---------|---------|
| `python -m src.data.collector --local` | Collect data locally |
| `python -m src.data.parser <input> --output <dir>` | Parse .pb files |
| `python -m src.data.features <input> --output <file>` | Engineer features |
| `python scripts/train_model.py` | Train model end-to-end |
| `python -m src.api.main` | Run prediction API |

## File Locations

| Type | Location |
|------|----------|
| Raw data | `data/raw/gtfs_rt/trip_updates/` |
| Processed data | `data/processed/` |
| Features | `data/features/` |
| Saved models | `artifacts/models/` |
| Metrics | `artifacts/metrics/` |
| Visualizations | `artifacts/visualizations/` |
