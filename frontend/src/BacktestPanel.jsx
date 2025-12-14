import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const BacktestPanel = ({ ticker, config }) => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [buyThresh, setBuyThresh] = useState(70);
  const [sellThresh, setSellThresh] = useState(30);

  const runSimulation = async () => {
    setLoading(true);
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/backtest', {
        ticker, config, buyThreshold: parseFloat(buyThresh), sellThreshold: parseFloat(sellThresh)
      });
      setResults(response.data);
    } catch (err) { alert("Błąd symulacji"); } finally { setLoading(false); }
  };

  return (
    <div>
      <div className="controls" style={{ justifyContent: 'flex-start', marginBottom: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <label>Kup &ge;</label><input type="number" value={buyThresh} onChange={e => setBuyThresh(e.target.value)} style={{width: '60px'}} />
          <label>Sprzedaj &le;</label><input type="number" value={sellThresh} onChange={e => setSellThresh(e.target.value)} style={{width: '60px'}} />
          <button onClick={runSimulation} disabled={loading}>{loading ? 'Liczenie...' : 'Symuluj'}</button>
        </div>
      </div>
      {results && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '20px' }}>
          <div className="stat-box"><div className="stat-label">Twój Wynik</div><div className={`stat-val ${results.total_return >= 0 ? 'green' : 'red'}`}>{results.total_return}%</div><div className="stat-sub">{results.final_value} USD</div></div>
          <div className="stat-box"><div className="stat-label">Buy & Hold</div><div className="stat-val">{results.buy_and_hold_return}%</div><div className="stat-sub">Rynek</div></div>
          <div className="stat-box"><div className="stat-label">Transakcje</div><div className="stat-val">{results.trades_count}</div><div className="stat-sub">Liczba</div></div>
        </div>
      )}
    </div>
  );
};

export default BacktestPanel;