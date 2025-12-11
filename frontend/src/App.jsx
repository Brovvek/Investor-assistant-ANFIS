// src/App.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import MarketChart from './MarketChart';
import './App.css';

function App() {
  const [ticker, setTicker] = useState('^GSPC');
  const [chartData, setChartData] = useState(null);
  const [tickersList, setTickersList] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // 1. Pobierz listę tickerów przy starcie
  useEffect(() => {
    axios.get('http://127.0.0.1:5000/api/tickers')
      .then(res => setTickersList(res.data))
      .catch(err => console.error("Błąd pobierania tickerów:", err));
      
    // Automatyczne uruchomienie dla domyślnego tickera
    handleAnalysis('^GSPC');
  }, []);

  // 2. Funkcja analizy
  const handleAnalysis = async (symbolToAnalyze) => {
    const symbol = symbolToAnalyze || ticker;
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/analyze', {
        ticker: symbol
      });
      setChartData(response.data);
    } catch (err) {
      setError(err.response?.data?.error || "Błąd połączenia z serwerem");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="header">
        <div className="title">
          Investor Assistant <span className="highlight">ANFIS</span> React
        </div>
      </div>

      <div className="controls">
        <input 
          type="text" 
          value={ticker} 
          onChange={(e) => setTicker(e.target.value)}
          placeholder="Wpisz symbol (np. ^NDX)" 
          list="ticker-options"
        />
        <datalist id="ticker-options">
          {tickersList.map((t) => (
            <option key={t.symbol} value={t.symbol}>
              {t.name}
            </option>
          ))}
        </datalist>

        <button onClick={() => handleAnalysis()} disabled={loading}>
          {loading ? 'Przetwarzanie...' : 'Analizuj'}
        </button>
      </div>

      {error && <div style={{color: 'red', textAlign: 'center', marginBottom: 10}}>{error}</div>}

      <div className="chart-container">
        <MarketChart data={chartData} />
      </div>
    </div>
  );
}

export default App;