import React, { useState, useEffect } from 'react';
import axios from 'axios';
import MarketChart from './MarketChart';
import InfoBadges from './InfoBadges';
import CorrelationPanel from './CorrelationPanel';
import BacktestPanel from './BacktestPanel';
import CollapsibleSection from './CollapsibleSection';
import AnfisMLPanel from './AnfisMLPanel';
import './App.css';

function App() {
  const [ticker, setTicker] = useState('BTC-USD');
  const [chartData, setChartData] = useState(null);
  const [tickersList, setTickersList] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [timeWindow, setTimeWindow] = useState(1260);  // Default: 1260 days (~5 years)
  
  // Domyślnie pusta konfiguracja - wymuszamy wybór strategii
  const [indicatorsConfig, setIndicatorsConfig] = useState({});  

  useEffect(() => {
    axios.get('http://127.0.0.1:5000/api/tickers').then(res => setTickersList(res.data));
  }, []);

  const handleAnalysis = async (symbolToAnalyze) => {
    const symbol = symbolToAnalyze || ticker;
    setLoading(true); setError('');
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/analyze', { ticker: symbol, config: indicatorsConfig, timeWindow: timeWindow });
      setChartData(response.data);
    } catch (err) { setError(err.response?.data?.error || "Błąd serwera"); } finally { setLoading(false); }
  };

  return (
    <div className="container">
      <div className="header">
        <div className="title">
          Investor Assistant
        </div>
      </div>

      {/* 1. GŁÓWNY PANEL STEROWANIA */}
      <div className="main-controls-box">
        <InfoBadges 
          config={indicatorsConfig} 
          onConfigChange={setIndicatorsConfig}
          ticker={ticker} 
        />
        
        <div className="controls-row">
          <input 
            type="text" 
            value={ticker} 
            onChange={(e) => setTicker(e.target.value)}
            placeholder="Symbol (np. ^NDX)" 
            list="ticker-options"
          />
          <datalist id="ticker-options">
            {tickersList.map((t) => <option key={t.symbol} value={t.symbol}>{t.name}</option>)}
          </datalist>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <label htmlFor="time-window" style={{ whiteSpace: 'nowrap' }}>Okres:</label>
            <select 
              id="time-window"
              value={timeWindow} 
              onChange={(e) => setTimeWindow(parseInt(e.target.value))}
              style={{ padding: '8px', borderRadius: '4px', border: '1px solid #ccc' }}
            >
              <option value={30}>Ostatnie 30 dni (1 miesiąc)</option>
              <option value={90}>Ostatnie 90 dni (3 miesiące)</option>
              <option value={126}>Ostatnie 126 dni (6 miesięcy)</option>
              <option value={252}>Ostatnie 252 dni (1 rok)</option>
              <option value={504}>Ostatnie 504 dni (2 lata)</option>
              <option value={756}>Ostatnie 756 dni (3 lata)</option>
              <option value={1260}>Ostatnie 1260 dni (5 lat)</option>
              <option value={1890}>Ostatnie 1890 dni (7 lat)</option>
              <option value={2520}>Ostatnie 2520 dni (10 lat)</option>
              <option value={3780}>Ostatnie 3780 dni (15 lat)</option>
              <option value={5040}>Ostatnie 5040 dni (20 lat)</option>
              <option value={99999}>Wszystkie dostępne dane</option>
            </select>
          </div>
          
          <button onClick={() => handleAnalysis()} disabled={loading}>
            {loading ? 'Przeliczanie...' : 'Analizuj Rynek'}
          </button>
        </div>
        {error && <div className="error-message">{error}</div>}
      </div>

      {/* PANEL KORELACJI (TERAZ PRZED WYKRESEM) */}
      <CollapsibleSection title="📊 Mapa Korelacji (Data Science)" defaultOpen={false}>
        <CorrelationPanel ticker={ticker} />
      </CollapsibleSection>

      {/* GŁÓWNY WYKRES */}
      <CollapsibleSection title="📈 Wykres Analityczny (Cena + FIS)" defaultOpen={true}>
        <div className="chart-container-600"><MarketChart data={chartData} /></div>
      </CollapsibleSection>

      {/* BACKTEST */}
      <CollapsibleSection title="💰 Symulator Strategii FIS(Backtest)" defaultOpen={false}>
        <BacktestPanel ticker={ticker} config={indicatorsConfig} timeWindow={timeWindow} />
      </CollapsibleSection>
      
      {/* UCZENIE ANFIS */}
      <CollapsibleSection title="🧠 Uczenie ANFIS (Deep Learning Prediction)" defaultOpen={false}>
        <AnfisMLPanel ticker={ticker} config={indicatorsConfig} />
      </CollapsibleSection>

    </div>
  );
}

export default App;
