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
  
  // Domyślnie pusta konfiguracja - wymuszamy wybór strategii
  const [indicatorsConfig, setIndicatorsConfig] = useState({}); 

  useEffect(() => {
    axios.get('http://127.0.0.1:5000/api/tickers').then(res => setTickersList(res.data));
  }, []);

  const handleAnalysis = async (symbolToAnalyze) => {
    const symbol = symbolToAnalyze || ticker;
    setLoading(true); setError('');
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/analyze', { ticker: symbol, config: indicatorsConfig });
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
        <BacktestPanel ticker={ticker} config={indicatorsConfig} />
      </CollapsibleSection>
      
      {/* UCZENIE ANFIS */}
      <CollapsibleSection title="🧠 Uczenie ANFIS (Deep Learning Prediction)" defaultOpen={false}>
        <AnfisMLPanel ticker={ticker} config={indicatorsConfig} />
      </CollapsibleSection>

    </div>
  );
}

export default App;
