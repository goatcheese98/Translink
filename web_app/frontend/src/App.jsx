import { useState } from 'react'
import './App.css'

const API_BASE = 'http://localhost:5000/api'

function App() {
  const [stopId, setStopId] = useState('')
  const [routeId, setRouteId] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handlePredict = async (e) => {
    e.preventDefault()

    if (!stopId || !routeId) {
      setError('Please enter both Stop ID and Route ID')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          stop_id: stopId,
          route_id: routeId,
        }),
      })

      if (!response.ok) {
        throw new Error('Prediction failed')
      }

      const data = await response.json()
      setPrediction(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const getRiskColor = (level) => {
    switch (level) {
      case 'high': return '#ef4444'
      case 'moderate': return '#f59e0b'
      case 'low': return '#10b981'
      default: return '#6b7280'
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>🚌 TransLink Delay Predictor</h1>
        <p className="tagline">Know before you go.</p>
      </header>

      <main className="main-content">
        <div className="predictor-card">
          <form onSubmit={handlePredict} className="input-form">
            <div className="form-group">
              <label htmlFor="stopId">Stop ID</label>
              <input
                type="text"
                id="stopId"
                placeholder="e.g., 50001"
                value={stopId}
                onChange={(e) => setStopId(e.target.value)}
                className="input"
              />
            </div>

            <div className="form-group">
              <label htmlFor="routeId">Route ID</label>
              <input
                type="text"
                id="routeId"
                placeholder="e.g., 099"
                value={routeId}
                onChange={(e) => setRouteId(e.target.value)}
                className="input"
              />
            </div>

            <button
              type="submit"
              className="predict-button"
              disabled={loading}
            >
              {loading ? 'Predicting...' : 'Check Delay Risk'}
            </button>
          </form>

          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}

          {prediction && (
            <div className="prediction-result">
              <div
                className="risk-indicator"
                style={{ backgroundColor: getRiskColor(prediction.risk_level) }}
              >
                <span className="risk-emoji">{prediction.emoji}</span>
                <h2 className="risk-level">
                  {prediction.risk_level.toUpperCase()} RISK
                </h2>
                <p className="probability">
                  Delay Probability: {(prediction.probability * 100).toFixed(1)}%
                </p>
              </div>

              <div className="details">
                <div className="detail-card">
                  <h3>💡 Recommendation</h3>
                  <p>{prediction.recommendation}</p>
                </div>

                <div className="detail-card">
                  <h3>📊 Route Statistics</h3>
                  <ul>
                    <li>Average Delay: {prediction.route_stats.avg_delay_min} min</li>
                    <li>Delay Rate: {(prediction.route_stats.delay_rate * 100).toFixed(1)}%</li>
                    <li>Trips Analyzed: {prediction.route_stats.trips_analyzed}</li>
                  </ul>
                </div>

                <div className="detail-card">
                  <h3>📍 Stop Statistics</h3>
                  <ul>
                    <li>Average Delay: {prediction.stop_stats.avg_delay_min} min</li>
                    <li>Delay Rate: {(prediction.stop_stats.delay_rate * 100).toFixed(1)}%</li>
                    <li>Trips Analyzed: {prediction.stop_stats.trips_analyzed}</li>
                  </ul>
                </div>

                {prediction.current_bus_status.live_data_available && (
                  <div className="detail-card live-status">
                    <h3>🔴 Live Status</h3>
                    <p>Current Delay: {prediction.current_bus_status.current_delay_min} min</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        <footer className="disclaimer">
          <p>⚠️ Beta Version - Predictions based on limited historical data. Always check TransLink's official app for real-time updates.</p>
        </footer>
      </main>
    </div>
  )
}

export default App
