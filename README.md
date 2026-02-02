# TransLink Delay Predictor

Real-time transit delay prediction for Vancouver's TransLink bus network using machine learning.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project predicts bus delays ≥10 minutes for TransLink routes in Vancouver, BC. It uses:

- **Real-time GTFS-RT data** from TransLink API
- **Logistic Regression** model with automatic retraining
- **Serverless GCP infrastructure** for $0 cost
- **28-day rolling data retention** with infinite trend aggregation

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd transit-delay-prediction
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r config/requirements.txt

# 3. Configure environment
cp config/.env.example .env
# Edit .env with your TransLink API key

# 4. Run data collection
python -m src.data.collector

# 5. Parse data
python -m src.data.parser

# 6. Train model
python -m src.models.train

# 7. Start API
python -m src.api.main
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

```
Data Collection (Cloud Function) → BigQuery → Training → Cloud Run API
        ↑                              ↓
   (Every 5 min)              (Weekly retraining)
```

## Project Structure

```
transit-delay-prediction/
├── src/                    # Source code
│   ├── data/              # Data pipeline
│   │   ├── collector.py   # GTFS-RT scraper
│   │   ├── parser.py      # Parse .pb files
│   │   └── features.py    # Feature engineering
│   ├── models/            # ML code
│   │   ├── train.py       # Training pipeline
│   │   ├── evaluate.py    # Model evaluation
│   │   └── predict.py     # Prediction logic
│   └── api/               # Prediction API
│       └── main.py        # FastAPI app
├── notebooks/             # Exploration notebooks
├── artifacts/             # ML artifacts
│   ├── models/           # Saved models
│   └── metrics/          # Evaluation results
├── data/                  # Data directory
│   ├── raw/              # Raw .pb files
│   └── processed/        # Cleaned CSVs
├── config/                # Configuration
│   ├── requirements.txt
│   └── .env.example
├── docs/                  # Documentation
├── tests/                 # Unit & integration tests
└── infrastructure/        # GCP deployment scripts
```

## Key Features

- **Real-time predictions**: Sub-300ms latency via Cloud Run
- **Automatic retraining**: Weekly pipeline with conditional deployment
- **Cost-optimized**: $0/month within GCP free tiers
- **Smart retention**: 4 weeks granular data + infinite aggregated trends

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — System design & GCP setup
- [DATA_COLLECTION_PLAN.md](DATA_COLLECTION_PLAN.md) — Data collection strategy

## Development

```bash
# Install dev dependencies
pip install -r config/requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/ tests/
isort src/ tests/

# Type check
mypy src/
```

## License

MIT License — see LICENSE file.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

---

**Note**: This project is not affiliated with TransLink. It uses their public GTFS-RT API.
