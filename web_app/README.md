# TransLink Delay Predictor Web App

A real-time web application that predicts transit delays for Vancouver's TransLink bus system.

## 🚀 Quick Start

### **Backend (Flask API)**

1. Navigate to backend directory:

   ```bash
   cd web_app/backend
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (optional, for live data):

   ```bash
   # Create .env file in project root if you want live GTFS-RT data
   echo "TRANSLINK_API_KEY=your_api_key_here" > ../../.env
   ```

4. Start the Flask server:

   ```bash
   python app.py
   ```

   The API will run on `http://localhost:5000`

### **Frontend (React)**

1. Navigate to frontend directory:

   ```bash
   cd web_app/frontend
   ```

2. Install dependencies (if not done automatically):

   ```bash
   npm install
   ```

3. Start the development server:

   ```bash
   npm run dev
   ```

   The app will run on `http://localhost:5173`

## 📖 How to Use

1. **Open the app** at `http://localhost:5173`
2. **Enter a Stop ID** (e.g., `50001`)
3. **Enter a Route ID** (e.g., `099`)
4. **Click "Check Delay Risk"**
5. **View the prediction**:
   - 🟢 Low Risk = Bus expected on time
   - 🟡 Moderate Risk = 5-10 min delay possible
   - 🔴 High Risk = Major delay likely (10+ min)

## 🔑 Finding Stop/Route IDs

### Method 1: From Our Data

Check the exported artifacts:

```bash
cat ../artifacts/route_risk_factors.json | grep route_id
cat ../artifacts/stop_risk_factors.json | grep stop_id
```

### Method 2: TransLink GTFS Static Data

Look in `data/raw/gtfs_static/`:

- `routes.txt` - All route IDs
- `stops.txt` - All stop IDs

### Example Valid IDs (from our dataset)

- **Routes:** 099, 003, 014, 080, 010
- **Stops:** Check the artifacts folder for actual stop IDs from the dataset

## 🏗️ Architecture

```
web_app/
├── backend/           # Flask API
│   ├── app.py        # Main API server
│   └── requirements.txt
├── frontend/          # React UI
│   ├── src/
│   │   ├── App.jsx   # Main component
│   │   └── App.css   # Styling
│   └── package.json
└── artifacts/         # Model & data
    ├── model.pkl
    ├── scaler.pkl
    ├── route_risk_factors.json
    └── stop_risk_factors.json
```

## 🔌 API Endpoints

### `POST /api/predict`

Main prediction endpoint.

**Request:**

```json
{
  "stop_id": "50001",
  "route_id": "099"
}
```

**Response:**

```json
{
  "risk_level": "low",
  "emoji": "🟢",
  "probability": 0.123,
  "recommendation": "✅ Low delay risk. Bus expected on time.",
  "route_stats": {
    "avg_delay_min": 2.1,
    "delay_rate": 0.085
  },
  "stop_stats": {
    "avg_delay_min": 1.8,
    "delay_rate": 0.072
  }
}
```

### `GET /api/health`

Health check.

### `GET /api/stops`

Get all available stops.

### `GET /api/routes`

Get all available routes.

### `GET /api/route-stats/<route_id>`

Get historical stats for a specific route.

## ⚠️ Current Limitations

- **Limited Training Data:** Model trained on single snapshot (Jan 17, 12:47 AM)
- **Time Generalization:** Predictions may vary for different times of day
- **Route/Stop Coverage:** Only includes routes/stops from the training snapshot

## 🛠️ Troubleshooting

### Backend won't start

```bash
# Make sure you're in the right directory
cd web_app/backend

# Check that artifacts exist
ls ../artifacts/

# Re-export artifacts if needed
cd ../..
python scripts/export_model_artifacts.py
```

### Frontend shows CORS errors

Make sure the Flask backend is running on `http://localhost:5000` and has `flask-cors` installed.

### "Route/Stop not found"

Use IDs from the training dataset. Check `artifacts/route_risk_factors.json` and `artifacts/stop_risk_factors.json` for valid IDs.

## 🎨 Tech Stack

- **Frontend:** React, Vite
- **Backend:** Flask, Python
- **ML:** Scikit-learn (Logistic Regression)
- **Data:** TransLink GTFS Static & Real-Time

## 📈 Next Steps

- [ ] Collect more temporal data (rush hour, weekends)
- [ ] Add stop/route autocomplete with names
- [ ] Integrate map visualization
- [ ] Deploy to production (Vercel + Railway)
- [ ] Add alternative route suggestions

## 👥 Team

Built for the TransLink Delay Prediction project.

---

**Made with ❤️ and data science**
