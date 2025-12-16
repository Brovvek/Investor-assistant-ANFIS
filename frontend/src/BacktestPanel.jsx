import React, { useState } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js'; // <--- Importujemy wykres
import './App.css';

const BacktestPanel = ({ ticker, config }) => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // Domyślne parametry (bezpieczniejsze)
  const [buyThresh, setBuyThresh] = useState(65);
  const [sellThresh, setSellThresh] = useState(35);
  const [stopLoss, setStopLoss] = useState(5);
  const [takeProfit, setTakeProfit] = useState(0);
  const [trailingStop, setTrailingStop] = useState(false);

  const runSimulation = async () => {
    setLoading(true);
    setResults(null); // Reset wyników
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/backtest', {
        ticker, config,
        buyThreshold: parseFloat(buyThresh),
        sellThreshold: parseFloat(sellThresh),
        stopLoss: parseFloat(stopLoss),
        takeProfit: parseFloat(takeProfit),
        trailingStop: trailingStop
      });
      setResults(response.data);
    } catch (err) { 
        console.error(err);
        alert("Błąd symulacji. Sprawdź czy pobrano dane w sekcji Analizy."); 
    } finally { 
        setLoading(false); 
    }
  };

  return (
    <div style={{ padding: '10px' }}>
      
      {/* 1. KONTROLKI STRATEGII */}
      <div className="controls-row" style={{ flexWrap: 'wrap', gap: '20px', alignItems: 'flex-end', borderTop: 'none', paddingTop: 0, justifyContent: 'center' }}>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
          <span style={{fontSize: '0.8em', color: '#888'}}>Sygnały ANFIS (0-100)</span>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <label>Kup &ge;</label>
            <input type="number" value={buyThresh} onChange={e => setBuyThresh(e.target.value)} style={{width: '50px'}} />
            <label>Sprzedaj &le;</label>
            <input type="number" value={sellThresh} onChange={e => setSellThresh(e.target.value)} style={{width: '50px'}} />
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
          <span style={{fontSize: '0.8em', color: '#888'}}>Zarządzanie Ryzykiem (%)</span>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <label>Stop Loss</label>
            <input type="number" value={stopLoss} onChange={e => setStopLoss(e.target.value)} style={{width: '50px'}} />
            <label>Take Profit</label>
            <input type="number" value={takeProfit} onChange={e => setTakeProfit(e.target.value)} style={{width: '50px'}} />
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
             <span style={{fontSize: '0.8em', color: '#888'}}>Opcje</span>
             <label style={{display: 'flex', alignItems: 'center', gap: '5px', cursor: 'pointer', height: '30px'}}>
                <input type="checkbox" checked={trailingStop} onChange={e => setTrailingStop(e.target.checked)} />
                Trailing Stop
             </label>
        </div>

        <button onClick={runSimulation} disabled={loading} style={{ height: '35px', padding: '0 20px' }}>
            {loading ? 'Symulacja...' : '▶ Uruchom Backtest'}
        </button>
      </div>

      {/* 2. WYNIKI I WYKRES */}
      {results && (
        <div style={{ marginTop: '30px' }}>
            
            {/* A. STATYSTYKI W KAFELKACH */}
            <div className="stats-grid">
                <div className="stat-box" style={{borderColor: results.total_return > results.buy_and_hold_return ? '#00e676' : '#444'}}>
                    <div className="stat-label">Twoja Strategia</div>
                    <div className={`stat-val ${results.total_return >= 0 ? 'green' : 'red'}`}>
                        {results.total_return > 0 ? '+' : ''}{results.total_return}%
                    </div>
                    <div className="stat-sub">{results.final_value.toLocaleString()} USD</div>
                </div>
                
                <div className="stat-box">
                    <div className="stat-label">Rynek (Buy & Hold)</div>
                    <div className="stat-val" style={{color: '#89b4fa'}}>
                        {results.buy_and_hold_return > 0 ? '+' : ''}{results.buy_and_hold_return}%
                    </div>
                    <div className="stat-sub">Benchmark</div>
                </div>

                <div className="stat-box">
                    <div className="stat-label">Sharpe Ratio</div>
                    <div className={`stat-val ${results.sharpe_ratio >= 1.0 ? 'green' : (results.sharpe_ratio < 0.5 ? 'red' : '')}`}>
                        {results.sharpe_ratio}
                    </div>
                    <div className="stat-sub">Ryzyko/Zysk</div>
                </div>

                <div className="stat-box">
                    <div className="stat-label">Max Drawdown</div>
                    <div className="stat-val red">-{results.max_drawdown}%</div>
                    <div className="stat-sub">Maks. Obsunięcie</div>
                </div>
            </div>

            {/* B. WYKRES KRZYWEJ KAPITAŁU (NOWOŚĆ) */}
            <div style={{ marginTop: '20px', height: '400px', border: '1px solid #363a45', borderRadius: '8px', padding: '10px', background: '#1e222d' }}>
                 <h4 style={{margin: '0 0 10px 0', textAlign: 'center', color: '#888'}}>Porównanie Wyników (Kapitał Startowy: {results.initial_capital}$)</h4>
                 <Plot
                    data={[
                        {
                            x: results.chart_dates,
                            y: results.bnh_curve,
                            type: 'scatter',
                            mode: 'lines',
                            name: 'Buy & Hold (Rynek)',
                            line: { color: 'rgba(255, 255, 255, 0.3)', dash: 'dot', width: 2 }
                        },
                        {
                            x: results.chart_dates,
                            y: results.equity_curve,
                            type: 'scatter',
                            mode: 'lines',
                            name: 'Twoja Strategia',
                            line: { color: '#00e676', width: 3 },
                            fill: 'tozeroy',
                            fillcolor: 'rgba(0, 230, 118, 0.1)'
                        }
                    ]}
                    layout={{
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: '#d1d4dc' },
                        margin: { t: 20, b: 40, l: 60, r: 20 },
                        xaxis: { showgrid: false },
                        yaxis: { title: 'Wartość Portfela ($)', gridcolor: '#2a2e39' },
                        legend: { orientation: 'h', y: 1.1, x: 0.5, xanchor: 'center' }
                    }}
                    useResizeHandler={true}
                    style={{ width: "100%", height: "100%" }}
                 />
            </div>

            {/* C. LISTA TRANSAKCJI */}
            <div style={{ marginTop: '20px', maxHeight: '300px', overflowY: 'auto', background: '#1a1d26', padding: '15px', borderRadius: '8px', fontSize: '0.85em' }}>
                <h4 style={{marginTop: 0, marginBottom: '10px', borderBottom: '1px solid #333', paddingBottom: '5px'}}>Ostatnie Transakcje</h4>
                <table style={{width: '100%', borderCollapse: 'collapse', textAlign: 'left'}}>
                    <thead style={{position: 'sticky', top: 0, background: '#1a1d26'}}>
                        <tr style={{color: '#888'}}>
                            <th style={{padding:'5px'}}>Data</th>
                            <th>Akcja</th>
                            <th>Cena</th>
                            <th>Powód</th>
                            <th>Wynik</th>
                        </tr>
                    </thead>
                    <tbody>
                        {results.trades.slice().reverse().map((t, i) => (
                            <tr key={i} style={{borderBottom: '1px solid #2a2e39'}}>
                                <td style={{padding:'8px 5px'}}>{t.date}</td>
                                <td style={{color: t.type === 'BUY' ? '#00e676' : '#ff5252', fontWeight: 'bold'}}>{t.type}</td>
                                <td>{t.price.toFixed(2)}</td>
                                <td style={{fontStyle: 'italic', color: '#aaa'}}>{t.reason}</td>
                                <td>
                                    {t.profit_pct !== undefined ? (
                                        <span style={{color: t.profit_pct > 0 ? '#00e676' : '#ff5252', fontWeight: 'bold'}}>
                                            {t.profit_pct > 0 ? '+' : ''}{t.profit_pct.toFixed(2)}%
                                        </span>
                                    ) : '-'}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
      )}
    </div>
  );
};

export default BacktestPanel;