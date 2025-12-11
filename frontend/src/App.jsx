// src/App.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import MarketChart from './MarketChart';
import InfoBadges from './InfoBadges';
import './App.css';

function App() {
  const [ticker, setTicker] = useState('^GSPC');
  const [chartData, setChartData] = useState(null);
  const [tickersList, setTickersList] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // NOWOŚĆ: Stan konfiguracji wag i aktywności
  const [indicatorsConfig, setIndicatorsConfig] = useState({
    rsi:   { enabled: true, weight: 1.0 },
    vix:   { enabled: true, weight: 1.0 },
    yield: { enabled: true, weight: 1.0 }
  });

  // Pobieranie tickerów (bez zmian)
  useEffect(() => {
    axios.get('http://127.0.0.1:5000/api/tickers')
      .then(res => setTickersList(res.data))
      .catch(err => console.error(err));
  }, []);

  // Analiza po zmianie konfiguracji lub tickera
  const handleAnalysis = async (symbolToAnalyze) => {
    const symbol = symbolToAnalyze || ticker;
    setLoading(true);
    setError('');
    
    try {
      // NOWOŚĆ: Wysyłamy 'config' do backendu
      const response = await axios.post('http://127.0.0.1:5000/api/analyze', {
        ticker: symbol,
        config: indicatorsConfig // Przekazujemy wagi
      });
      setChartData(response.data);
    } catch (err) {
      setError(err.response?.data?.error || "Błąd serwera");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="header">
        <div className="title">
          Investor Assistant <span className="highlight">ANFIS</span> Pro
        </div>
      </div>

      {/* Panel Sterowania Wagami */}
      <InfoBadges 
        config={indicatorsConfig} 
        onConfigChange={setIndicatorsConfig} 
      />

      <div className="controls">
        <input 
          type="text" 
          value={ticker} 
          onChange={(e) => setTicker(e.target.value)}
          placeholder="Symbol..." 
          list="ticker-options"
        />
        <datalist id="ticker-options">
          {tickersList.map((t) => <option key={t.symbol} value={t.symbol}>{t.name}</option>)}
        </datalist>

        <button onClick={() => handleAnalysis()} disabled={loading}>
          {loading ? 'Przeliczanie...' : 'Analizuj z wagami'}
        </button>
      </div>

      {error && <div style={{color: 'red', textAlign: 'center'}}>{error}</div>}

      <div className="chart-container">
        <MarketChart data={chartData} />
      </div>
    </div>
  );
}

export default App;