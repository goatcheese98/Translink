# TransLink Delay Predictor — Cloud Architecture

**Version:** 2.0  
**Date:** 2026-02-02  
**Goal:** Cost-optimized, auto-retraining ML system with 2-4 week data retention

---

## 🎯 Executive Summary

This document proposes a **serverless-first, event-driven architecture** using Google Cloud Platform (GCP). It prioritizes free tiers while ensuring production reliability and automatic model retraining.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Serverless-first** | Pay-per-use vs. always-on VMs; aligns with intermittent traffic |
| **BigQuery as feature store** | Free tier (1TB queries/month), SQL-native, time-travel built-in |
| **2-tier data retention** | Hot: 4 weeks granular; Cold: aggregated trends forever |
| **Weekly retraining** | Balances model freshness with cost |
| **Real-time predictions** | Sub-500ms latency via Cloud Run |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TRANSLINK DELAY PREDICTOR                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────────┐   │
│  │   Web App    │     │   Mobile     │     │       Admin Dashboard        │   │
│  │(Firebase)    │◄────┤   (Future)   │────►│    (Next.js + BigQuery)      │   │
│  └──────┬───────┘     └──────────────┘     └──────────────────────────────┘   │
│         │                                                                       │
│         │ GET /predict?stop_id=123&route_id=99                                  │
│         ▼                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐      │
│  │                     GCP Cloud Run (Prediction API)                    │      │
│  │  ┌─────────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │      │
│  │  │  FastAPI App    │  │ Model Cache │  │  Feature Lookup (BQ)    │  │      │
│  │  │  • Load model   │  │ • Hot model │  │  • Real-time features   │  │      │
│  │  │  • Fetch feats  │  │ • LRU cache │  │  • 4-week rolling       │  │      │
│  │  │  • Predict      │  │ • <100MB    │  │  • Historical trends    │  │      │
│  │  └─────────────────┘  └─────────────┘  └─────────────────────────┘  │      │
│  └──────────────────────────────────────────────────────────────────────┘      │
│                                    ▲                                            │
│                                    │                                            │
│  ┌─────────────────────────────────┴─────────────────────────────────────┐     │
│  │                         DATA PLATFORM (GCP)                            │     │
│  │                                                                        │     │
│  │  ┌──────────────┐    ┌──────────────┐    ┌────────────────────────┐   │     │
│  │  │   INGESTION  │    │  PROCESSING  │    │      STORAGE LAYER     │   │     │
│  │  │              │    │              │    │                        │   │     │
│  │  │ Cloud Scheduler│   │ Cloud Functions│   │  ┌────────────────┐   │   │     │
│  │  │ (every 5 min) │──►│  • Parse PB    │──►│  │ GCS Raw Data   │   │   │     │
│  │  │              │    │  • Validate    │    │  │  (4 weeks TTL) │   │   │     │
│  │  │ Cloud Function│   │  • Transform   │    │  └────────────────┘   │   │     │
│  │  │ (API scraper) │   │              │    │                        │   │     │
│  │  └──────────────┘    └──────┬───────┘    │  ┌────────────────┐   │   │     │
│  │                             │            │  │ BigQuery       │   │   │     │
│  │                             ▼            │  │ • Features     │   │   │     │
│  │                      ┌──────────────┐    │  │ • Aggregates   │   │   │     │
│  │                      │  Pub/Sub     │    │  │ • Trends       │   │   │     │
│  │                      │  (events)    │    │  │ (infinite)     │   │   │     │
│  │                      └──────────────┘    │  └────────────────┘   │   │     │
│  │                             │            │                        │   │     │
│  │                             ▼            │  ┌────────────────┐   │   │     │
│  │  ┌────────────────────────────────────┐  │  │ Artifact Reg.  │   │   │     │
│  │  │      ML PIPELINE                   │  │  │ • Models       │   │   │     │
│  │  │                                    │  │  │ • Metadata     │   │   │     │
│  │  │  Cloud Scheduler (weekly)          │  │  └────────────────┘   │   │     │
│  │  │           │                        │  │                        │   │     │
│  │  │           ▼                        │  └────────────────────────┘   │     │
│  │  │  ┌────────────────────────┐        │                                │     │
│  │  │  │ Cloud Function/Run Job│        │                                │     │
│  │  │  │ • Feature engineering │        │                                │     │
│  │  │  │ • Train model         │        │                                │     │
│  │  │  │ • Evaluate vs previous│        │                                │     │
│  │  │  │ • Conditional deploy  │        │                                │     │
│  │  │  └────────────────────────┘        │                                │     │
│  │  │                                    │                                │     │
│  │  │  Deployment: If F1 > previous      │                                │     │
│  │  └────────────────────────────────────┘                                │     │
│  │                                                                        │     │
│  └────────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Flow: Ingestion to Prediction

### 1. Data Ingestion (Every 5 Minutes)

```yaml
Trigger: Cloud Scheduler (cron: */5 * * * *)
Action: HTTP POST to Cloud Function
Function: collect_gtfs_rt
  - Call TransLink GTFS-RT API
  - Save raw .pb to GCS: gs://translink-raw/trip_updates/{timestamp}.pb
  - Publish event to Pub/Sub: raw-data-ingested
Cost: ~$0 (Cloud Functions free: 2M invocations/month)
```

### 2. Data Processing (Event-Driven)

```yaml
Trigger: Pub/Sub message
Action: Cloud Function parse_and_load
  - Parse .pb → structured data
  - Validate schema + data quality
  - Load to BigQuery: staging.trip_updates
  - Aggregate to feature tables
Cost: ~$0 (Cloud Functions free tier)
```

### 3. Feature Engineering (Materialized Views)

BigQuery scheduled queries run every hour:

```sql
-- Rolling 4-week features (hot)
CREATE OR REPLACE MATERIALIZED VIEW ml_features.stop_features_4w AS
SELECT 
  stop_id,
  route_id,
  AVG(delay_min) as avg_delay_4w,
  STDDEV(delay_min) as std_delay_4w,
  AVG(CAST(delay_10plus AS FLOAT64)) as delay_rate_4w,
  COUNT(*) as trip_count_4w,
  MAX(feed_timestamp) as last_updated
FROM staging.trip_updates
WHERE feed_timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 28 DAY)
GROUP BY stop_id, route_id;

-- Historical trends (cold, aggregated)
CREATE OR REPLACE TABLE ml_features.historical_trends AS
SELECT 
  route_id,
  EXTRACT(HOUR FROM timestamp) as hour,
  EXTRACT(DOW FROM timestamp) as day_of_week,
  AVG(delay_min) as hist_avg_delay,
  AVG(CAST(delay_10plus AS FLOAT64)) as hist_delay_rate
FROM staging.trip_updates
GROUP BY route_id, hour, day_of_week;
```

---

## 🌐 Frontend Hosting

### Recommended: Firebase Hosting

| Service | Free Tier | Pros |
|---------|-----------|------|
| **Firebase Hosting** | 10GB/month | Same GCP project, zero config, automatic CDN |

**Why Firebase?**
- Same GCP project (unified billing)
- Zero-config CDN
- Free SSL, custom domains
- 10GB is plenty (your app likely <10MB)

```bash
# Deploy to Firebase
cd frontend
npm run build
firebase deploy --only hosting
```

### Alternatives

| Service | Free Tier | Notes |
|---------|-----------|-------|
| **Cloud Run** | 2M requests | Same infra as API, but cold starts |
| **Netlify** | 100GB/month | Git-based CI/CD |
| **Cloudflare Pages** | Unlimited | Fastest global CDN |

---

## 🤖 Automated Training Options

### Option Comparison

| Option | Best For | Cost | Complexity |
|--------|----------|------|------------|
| **Cloud Functions** (Recommended) | Logistic regression, starting out | **$0** | Low |
| **Cloud Run Jobs** | XGBoost, >10min training | ~$1-5/run | Medium |
| **Vertex AI Training** | GPU, hyperparameter tuning | ~$0.50-2/run | Medium |
| **Vertex AI Pipelines** | Full orchestration, team | ~$1-3/run | High |

### Recommended: Cloud Functions (Tier 1)

```python
# training_cloud_function.py
import functions_framework
import pickle
import json
from google.cloud import bigquery, storage
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd

PROJECT_ID = "your-project"
DATASET = "ml_features"
MODEL_BUCKET = "translink-models"

@functions_framework.cloud_event
def retrain_model(cloud_event):
    """Triggered by Cloud Scheduler every Sunday at 2 AM"""
    print("Starting model retraining...")
    
    # 1. Load data from BigQuery
    client = bigquery.Client()
    query = f"""
    SELECT 
      hour, day_of_week, is_weekend, is_rush_hour,
      route_avg_delay, route_std_delay, route_delay_rate,
      stop_avg_delay, stop_delay_rate, stop_sequence,
      delay_10plus as label
    FROM {PROJECT_ID}.{DATASET}.training_data
    WHERE feed_timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 28 DAY)
    ORDER BY feed_timestamp
    """
    df = client.query(query).to_dataframe()
    
    # 2. Split (80/20, time-based)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    feature_cols = [c for c in df.columns if c != 'label']
    X_train, y_train = train_df[feature_cols], train_df['label']
    X_test, y_test = test_df[feature_cols], test_df['label']
    
    # 3. Train model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # 4. Evaluate
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'f1_score': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'n_train': len(train_df),
        'n_test': len(test_df),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # 5. Load previous model metrics
    storage_client = storage.Client()
    bucket = storage_client.bucket(MODEL_BUCKET)
    
    try:
        prev_metrics_blob = bucket.blob('production/metrics.json')
        prev_metrics = json.loads(prev_metrics_blob.download_as_string())
        prev_f1 = prev_metrics.get('f1_score', 0)
    except:
        prev_f1 = 0
    
    # 6. Conditional deploy
    if metrics['f1_score'] > prev_f1 - 0.05:  # Allow 5% regression
        print(f"Deploying new model (F1: {metrics['f1_score']:.4f} > {prev_f1:.4f})")
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_cols,
            'metrics': metrics
        }
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # Save to candidates/
        candidate_blob = bucket.blob(f'candidates/model_{timestamp}.pkl')
        candidate_blob.upload_from_string(pickle.dumps(model_data))
        
        # Promote to production/
        prod_blob = bucket.blob('production/model.pkl')
        prod_blob.upload_from_string(pickle.dumps(model_data))
        
        # Save metrics
        metrics_blob = bucket.blob('production/metrics.json')
        metrics_blob.upload_from_string(json.dumps(metrics))
        
        print(f"✅ Model deployed! F1: {metrics['f1_score']:.4f}")
        return f"Deployed model with F1={metrics['f1_score']:.4f}", 200
    else:
        print(f"❌ Model rejected (F1: {metrics['f1_score']:.4f} < {prev_f1:.4f})")
        return f"Model rejected F1={metrics['f1_score']:.4f}", 200
```

**Deploy:**

```bash
# Deploy Cloud Function
gcloud functions deploy retrain_model \
  --runtime python311 \
  --trigger-topic retrain-trigger \
  --memory 1GB \
  --timeout 540s \
  --entry-point retrain_model

# Create Cloud Scheduler job
gcloud scheduler jobs create pubsub weekly-retrain \
  --schedule="0 2 * * 0" \
  --topic=retrain-trigger \
  --message-body="retrain"
```

---

## 🔄 Automatic Retraining Pipeline

### Weekly Flow

```
┌─────────────────┐
│ Cloud Scheduler │── Every Sunday 2:00 AM
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Cloud Function      │
│ (Training Script)   │
└────────┬────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ Step 1: Extract Features                     │
│   - Query BigQuery for last 28 days          │
│   - Join with historical trends              │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ Step 2: Train Model                          │
│   - Logistic Regression with class_weight    │
│   - StandardScaler for features              │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ Step 3: Evaluate                             │
│   - Calculate F1, Precision, Recall          │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│ Step 4: Conditional Deploy                   │
│   IF: New F1 > Current F1 - 0.05             │
│   THEN: Deploy to production/                │
│   ELSE: Save to failed/, keep current        │
└──────────────────────────────────────────────┘
```

### Model Versioning

```
gs://translink-models/
├── production/
│   ├── model.pkl          # Currently serving
│   └── metrics.json       # Current performance
├── candidates/
│   ├── model_20260201_020000.pkl
│   └── model_20260208_020000.pkl
└── failed/
    └── model_20260125_020000.pkl
```

---

## 🗂️ Data Retention & Lifecycle

### Tiered Storage

| Tier | Data Type | Retention | Cost |
|------|-----------|-----------|------|
| **Hot** | Raw .pb files | 28 days | Free |
| **Warm** | Parsed trip updates | 28 days | Free |
| **Cold** | Aggregated trends | Infinite | ~$0.02/GB/mo |
| **Archive** | Model artifacts | 90 days | ~$0.01/GB/mo |

### Automated Cleanup

```python
# Runs daily
import json

def lifecycle_config():
    return {
        "lifecycle": {
            "rule": [{
                "action": {"type": "Delete"},
                "condition": {
                    "age": 28,
                    "matchesPrefix": ["trip_updates/"]
                }
            }]
        }
    }

# Apply to GCS bucket
gsutil lifecycle set lifecycle.json gs://translink-raw
```

### BigQuery Partition Expiration

```sql
-- Raw data expires in 28 days
CREATE TABLE staging.trip_updates (
  -- ... columns ...
)
PARTITION BY DATE(feed_timestamp)
OPTIONS (partition_expiration_days = 28);

-- Aggregated trends kept forever
CREATE TABLE ml_features.route_trends (
  route_id STRING,
  date DATE,
  avg_delay FLOAT64,
  delay_rate FLOAT64
)
-- No expiration
```

---

## ⚡ Prediction Flow

### API Endpoint

```python
# FastAPI endpoint
@app.post("/predict")
async def predict_delay(request: PredictionRequest):
    """Target latency: <300ms"""
    
    # 1. Fetch features (cached)
    features = await fetch_features(
        stop_id=request.stop_id,
        route_id=request.route_id,
        hour=datetime.now().hour
    )
    
    # 2. Load model (cached in memory)
    model = get_cached_model()
    
    # 3. Predict
    prob = model.predict_proba([features])[0][1]
    
    return {
        "stop_id": request.stop_id,
        "route_id": request.route_id,
        "risk_level": "high" if prob > 0.7 else "moderate" if prob > 0.4 else "low",
        "probability": round(prob, 4),
        "features_used": {
            "stop_delay_rate_4w": features["stop_delay_rate"],
            "route_delay_rate_4w": features["route_delay_rate"],
        },
        "model_version": get_current_model_version(),
        "timestamp": datetime.utcnow().isoformat()
    }
```

### Caching Strategy

| Layer | TTL | Purpose |
|-------|-----|---------|
| **Cloud CDN** | 1 min | API responses |
| **In-memory** | Infinite | Model weights |

---

## 💰 Cost Estimates

### Free Tier Coverage

| Service | Free Tier | Your Usage | Cost |
|---------|-----------|------------|------|
| **Cloud Functions** | 2M invocations | ~9,000 | **$0** |
| **Cloud Run** | 2M requests | ~50K | **$0** |
| **Cloud Scheduler** | 3 jobs | 2 jobs | **$0** |
| **Pub/Sub** | 10GB | ~2GB | **$0** |
| **BigQuery** | 1TB queries, 10GB storage | ~500GB, 5GB | **$0** |
| **GCS** | 5GB | ~4GB | **$0** |
| **Firebase Hosting** | 10GB | ~100MB | **$0** |

**Total Expected: $0/month** (within free tiers)

### If Exceeding Free Tier

| Scenario | Monthly Cost |
|----------|--------------|
| Traffic 10× increase | ~$10-20 |
| Keep 8 weeks data | ~$5-10 |
| Add Redis cache | ~$35-50 |

---

## 🛠️ Implementation Phases

### Phase 1: MVP (Week 1-2)
- [ ] Set up GCP project with billing alerts
- [ ] Deploy Cloud Function for data collection
- [ ] Set up BigQuery tables
- [ ] Deploy prediction API to Cloud Run
- [ ] Deploy frontend to Firebase Hosting

### Phase 2: Auto-Retraining (Week 3-4)
- [ ] Build Cloud Function training pipeline
- [ ] Set up weekly retraining schedule
- [ ] Implement model comparison logic
- [ ] Add automated deployment

### Phase 3: Optimization (Week 5-6)
- [ ] Implement data lifecycle cleanup
- [ ] Add monitoring and alerting
- [ ] Load testing

### Phase 4: Advanced (Future)
- [ ] XGBoost ensemble model
- [ ] Weather data integration
- [ ] Push notifications

---

## 🔍 Monitoring

### Key Metrics

| Metric | Target | Alert If |
|--------|--------|----------|
| Prediction latency | <300ms | >500ms |
| F1 Score | >0.40 | <0.30 |
| Data freshness | <10 min | >15 min |
| API error rate | <1% | >5% |

---

## 📝 Summary

✅ **Cost:** $0/month within free tiers  
✅ **Frontend:** Firebase Hosting (GCP native)  
✅ **Training:** Cloud Functions (simple, serverless)  
✅ **Auto-Retraining:** Weekly with conditional deploy  
✅ **Retention:** 4 weeks granular + infinite trends  

---

## Next Steps

1. Set up GCP project
2. Deploy data collection Cloud Function
3. Test end-to-end flow
4. Build and deploy prediction API
