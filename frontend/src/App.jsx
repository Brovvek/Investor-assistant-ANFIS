import React, { useState, useEffect } from 'react';
import axios from 'axios';
import MarketChart from './MarketChart';
import InfoBadges from './InfoBadges';
import CorrelationPanel from './CorrelationPanel';
import BacktestPanel from './BacktestPanel';
import CollapsibleSection from './CollapsibleSection';
import './App.css';

function App() {
  const [ticker, setTicker] = useState('^GSPC');
  const [chartData, setChartData] = useState(null);
  const [tickersList, setTickersList] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  const [indicatorsConfig, setIndicatorsConfig] = useState({
    rsi:   { enabled: true, weight: 1.0 },
    vix:   { enabled: true, weight: 1.0 },
    yield: { enabled: true, weight: 1.0 },
    macd:  { enabled: true, weight: 1.0 },
    m2:    { enabled: true, weight: 1.0 }
  });

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
        <div className="title">Investor Assistant <span className="highlight">ANFIS</span> Pro</div>
      </div>

      <div style={{ background: '#1e222d', padding: '20px', borderRadius: '8px', marginBottom: '20px', border: '1px solid #363a45' }}>
        <InfoBadges config={indicatorsConfig} onConfigChange={setIndicatorsConfig} ticker={ticker} />
        <div className="controls" style={{ marginTop: '20px' }}>
          <input type="text" value={ticker} onChange={(e) => setTicker(e.target.value)} placeholder="Symbol (np. ^NDX)" list="ticker-options" />
          <datalist id="ticker-options">{tickersList.map((t) => <option key={t.symbol} value={t.symbol}>{t.name}</option>)}</datalist>
          <button onClick={() => handleAnalysis()} disabled={loading}>{loading ? 'Przeliczanie...' : 'Analizuj Rynek'}</button>
        </div>
        {error && <div style={{color: '#ef5350', textAlign: 'center', marginTop: '10px'}}>{error}</div>}
      </div>

      <CollapsibleSection title="Wykres Analityczny (Cena + ANFIS)" defaultOpen={true}>
        <div style={{ height: '600px' }}><MarketChart data={chartData} /></div>
      </CollapsibleSection>

      <CollapsibleSection title="Badanie Korelacji (Data Science)" defaultOpen={false}>
        <CorrelationPanel ticker={ticker} />
      </CollapsibleSection>

      <CollapsibleSection title="Symulator Strategii (Backtest)" defaultOpen={false}>
        <BacktestPanel ticker={ticker} config={indicatorsConfig} />
      </CollapsibleSection>
    </div>
  );
}

export default App;