"""
TransLink Delay Predictor - Flask API Backend

This is the main backend server that provides prediction endpoints
for the web application.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv
import requests
from google.transit import gtfs_realtime_pb2

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# ============================================================================
# LOAD MODEL ARTIFACTS ON STARTUP
# ============================================================================
print("Loading model artifacts...")

ARTIFACTS_DIR = Path("../artifacts")

# Load model and scaler
model = joblib.load(ARTIFACTS_DIR / "model.pkl")
scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")

# Load risk factors
with open(ARTIFACTS_DIR / "route_risk_factors.json") as f:
    route_risks = {r['route_id']: r for r in json.load(f)}

with open(ARTIFACTS_DIR / "stop_risk_factors.json") as f:
    stop_risks = {s['stop_id']: s for s in json.load(f)}

with open(ARTIFACTS_DIR / "global_defaults.json") as f:
    global_defaults = json.load(f)

with open(ARTIFACTS_DIR / "feature_names.json") as f:
    feature_names = json.load(f)

print(f"✅ Loaded model with {len(route_risks)} routes and {len(stop_risks)} stops")

# TransLink API Key
API_KEY = os.getenv("TRANSLINK_API_KEY")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fetch_live_gtfs_rt():
    """Fetch live GTFS-RT data from TransLink API"""
    if not API_KEY:
        return None
    
    try:
        url = f"https://gtfsapi.translink.ca/v3/gtfsrealtime?apikey={API_KEY}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.ParseFromString(response.content)
        return feed
    except Exception as e:
        print(f"Error fetching GTFS-RT: {e}")
        return None

def get_current_bus_delay(route_id, stop_id, feed):
    """Extract current delay for a specific route/stop from live feed"""
    if not feed:
        return 0
    
    for entity in feed.entity:
        if entity.HasField('trip_update'):
            trip = entity.trip_update
            if trip.trip.route_id == route_id:
                for stop in trip.stop_time_update:
                    if stop.stop_id == stop_id and stop.HasField('arrival'):
                        return stop.arrival.delay  # in seconds
    return 0

def build_feature_vector(stop_id, route_id, current_time, current_delay_sec=0):
    """Build feature vector for prediction"""
    # Get risk factors
    route_risk = route_risks.get(route_id, {
        'avg_delay': global_defaults['route_avg_delay'],
        'std_delay': 0,
        'delay_rate': global_defaults['route_delay_rate']
    })
    
    stop_risk = stop_risks.get(stop_id, {
        'avg_delay': global_defaults['stop_avg_delay'],
        'delay_rate': global_defaults['stop_delay_rate']
    })
    
    # Temporal features
    hour = current_time.hour
    day_of_week = current_time.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    is_rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18] else 0
    
    # Build feature array (must match training order)
    features = [
        hour,
        day_of_week,
        is_weekend,
        is_rush_hour,
        route_risk.get('avg_delay', 0),
        route_risk.get('std_delay', 0),
        route_risk.get('delay_rate', 0),
        stop_risk.get('avg_delay', 0),
        stop_risk.get('delay_rate', 0),
        0  # stop_sequence (we don't know this in advance, use 0)
    ]
    
    return np.array(features).reshape(1, -1)

def classify_risk(probability):
    """Convert probability to risk level"""
    if probability >= 0.7:
        return "high", "🔴"
    elif probability >= 0.4:
        return "moderate", "🟡"
    else:
        return "low", "🟢"

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "routes_loaded": len(route_risks),
        "stops_loaded": len(stop_risks)
    })

@app.route('/api/stops', methods=['GET'])
def get_stops():
    """Get all stops (for autocomplete)"""
    stops_list = [
        {"stop_id": stop_id, "delay_rate": data.get('delay_rate', 0)}
        for stop_id, data in stop_risks.items()
    ]
    # Sort by stop_id for now (would ideally have stop names)
    stops_list.sort(key=lambda x: x['stop_id'])
    return jsonify(stops_list[:100])  # Limit to 100 for demo

@app.route('/api/routes', methods=['GET'])
def get_routes():
    """Get all routes or routes for a specific stop"""
    stop_id = request.args.get('stop_id')
    
    # For now, return all routes (would need GTFS static to filter by stop)
    routes_list = [
        {"route_id": route_id, "delay_rate": data.get('delay_rate', 0)}
        for route_id, data in route_risks.items()
    ]
    routes_list.sort(key=lambda x: x['route_id'])
    return jsonify(routes_list)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        data = request.json
        stop_id = data.get('stop_id')
        route_id = data.get('route_id')
        departure_time = data.get('departure_time')  # Optional
        
        if not stop_id or not route_id:
            return jsonify({"error": "Missing stop_id or route_id"}), 400
        
        # Parse departure time or use current time
        if departure_time:
            current_time = datetime.fromisoformat(departure_time.replace('Z', '+00:00'))
        else:
            current_time = datetime.now()
        
        # Fetch live data
        live_feed = fetch_live_gtfs_rt()
        current_delay_sec = get_current_bus_delay(route_id, stop_id, live_feed)
        current_delay_min = current_delay_sec / 60
        
        # Build features
        features = build_feature_vector(stop_id, route_id, current_time, current_delay_sec)
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Classify risk
        risk_level, emoji = classify_risk(probability)
        
        # Get route/stop stats
        route_stats = route_risks.get(route_id, {})
        stop_stats = stop_risks.get(stop_id, {})
        
        # Generate recommendation
        if risk_level == "high":
            recommendation = "⚠️ High risk of major delay. Consider leaving 15 minutes early or choosing an alternative route."
        elif risk_level == "moderate":
            recommendation = "⚡ Moderate delay risk. Allow extra 5-10 minutes."
        else:
            recommendation = "✅ Low delay risk. Bus expected on time."
        
        # Response
        response = {
            "risk_level": risk_level,
            "emoji": emoji,
            "probability": round(probability, 3),
            "expected_delay_min": round(probability * 15, 1),  # Rough estimate
            "recommendation": recommendation,
            "current_bus_status": {
                "live_data_available": live_feed is not None,
                "current_delay_min": round(current_delay_min, 1) if live_feed else None
            },
            "route_stats": {
                "avg_delay_min": round(route_stats.get('avg_delay', 0), 1),
                "delay_rate": round(route_stats.get('delay_rate', 0), 3),
                "trips_analyzed": route_stats.get('trip_count', 0)
            },
            "stop_stats": {
                "avg_delay_min": round(stop_stats.get('avg_delay', 0), 1),
                "delay_rate": round(stop_stats.get('delay_rate', 0), 3),
                "trips_analyzed": stop_stats.get('trip_count', 0)
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/route-stats/<route_id>', methods=['GET'])
def route_stats(route_id):
    """Get historical stats for a specific route"""
    stats = route_risks.get(route_id, {})
    if not stats:
        return jsonify({"error": "Route not found"}), 404
    
    return jsonify({
        "route_id": route_id,
        "avg_delay_min": round(stats.get('avg_delay', 0), 1),
        "std_delay_min": round(stats.get('std_delay', 0), 1),
        "delay_rate": round(stats.get('delay_rate', 0), 3),
        "trips_analyzed": stats.get('trip_count', 0)
    })

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🚀 TransLink Delay Predictor API")
    print("=" * 70)
    print(f"Routes loaded: {len(route_risks)}")
    print(f"Stops loaded: {len(stop_risks)}")
    print(f"Live API: {'✅ Enabled' if API_KEY else '❌ Disabled (set TRANSLINK_API_KEY)'}")
    print("\nEndpoints:")
    print("  GET  /api/health")
    print("  GET  /api/stops")
    print("  GET  /api/routes")
    print("  POST /api/predict")
    print("  GET  /api/route-stats/<route_id>")
    print("\nStarting server on http://localhost:5000")
    print("=" * 70 + "\n")
    
    app.run(debug=True, port=5000)
