import React, { useState } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './App.css';

const BacktestPanel = ({ ticker, config }) => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const [buyThresh, setBuyThresh] = useState(60);
  const [sellThresh, setSellThresh] = useState(35);
  const [stopLoss, setStopLoss] = useState(100);
  const [takeProfit, setTakeProfit] = useState(0);
  const [trailingStop, setTrailingStop] = useState(false);

  const runSimulation = async () => {
    setLoading(true);
    setResults(null);
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

  // --- PRZYGOTOWANIE DANYCH DO WYKRESU ---
  let plotData = [];
  
  if (results && results.equity_curve && results.chart_dates) {
      
      // 1. Mapa pomocnicza: Data -> Wartość Portfela (Equity)
      const dateToEquity = {};
      results.chart_dates.forEach((date, index) => {
          dateToEquity[date] = results.equity_curve[index];
      });

      // 2. Przygotowanie markerów KUPNA i SPRZEDAŻY
      const buys = { x: [], y: [], text: [] };
      const sells = { x: [], y: [], text: [] };

      results.trades.forEach(trade => {
          const equityValue = dateToEquity[trade.date];
          if (equityValue === undefined) return;

          const tooltip = `Cena: ${trade.price.toFixed(2)}<br>Powód: ${trade.reason}`;

          if (trade.type === 'BUY') {
              buys.x.push(trade.date);
              buys.y.push(equityValue);
              buys.text.push(tooltip);
          } else if (trade.type === 'SELL') {
              sells.x.push(trade.date);
              sells.y.push(equityValue);
              sells.text.push(`${tooltip}<br>Wynik: ${trade.profit_pct.toFixed(2)}%`);
          }
      });

      // 3. Definicja serii danych (Traces)
      
      // Linia Benchmark (Rynek)
      plotData.push({
        x: results.chart_dates,
        y: results.bnh_curve,
        type: 'scatter',
        mode: 'lines',
        name: 'Buy & Hold (Rynek)',
        line: { color: 'rgba(255, 255, 255, 0.2)', dash: 'dot', width: 2 }
      });

      // Linia Strategii (Equity)
      plotData.push({
        x: results.chart_dates,
        y: results.equity_curve,
        type: 'scatter',
        mode: 'lines',
        name: 'Twoja Strategia',
        line: { color: '#00e676', width: 2 },
        fill: 'tozeroy',
        fillcolor: 'rgba(0, 230, 118, 0.05)'
      });

      // Markery KUPNA (Zielone trójkąty w górę)
      if (buys.x.length > 0) {
          plotData.push({
              x: buys.x,
              y: buys.y,
              mode: 'markers',
              name: 'Kupno',
              type: 'scatter',
              marker: { symbol: 'triangle-up', color: '#00e676', size: 10, line: { color: 'white', width: 1 } },
              text: buys.text,
              hoverinfo: 'text+x'
          });
      }

      // Markery SPRZEDAŻY (Czerwone trójkąty w dół)
      if (sells.x.length > 0) {
          plotData.push({
              x: sells.x,
              y: sells.y,
              mode: 'markers',
              name: 'Sprzedaż',
              type: 'scatter',
              marker: { symbol: 'triangle-down', color: '#ff5252', size: 10, line: { color: 'white', width: 1 } },
              text: sells.text,
              hoverinfo: 'text+x'
          });
      }
  }

  return (
    <div className="backtest-wrapper">
      
      {/* KONTROLKI STRATEGII */}
      <div className="controls-row backtest-controls">
        
        <div className="control-group">
          <span className="control-label">Sygnały FIS</span>
          <div className="control-inputs">
            <label>Kup &ge;</label>
            <input type="number" value={buyThresh} onChange={e => setBuyThresh(e.target.value)} className="input-small" />
            <label>Sprzedaj &le;</label>
            <input type="number" value={sellThresh} onChange={e => setSellThresh(e.target.value)} className="input-small" />
          </div>
        </div>

        <div className="control-group">
          <span className="control-label">Ryzyko (%)</span>
          <div className="control-inputs">
            <label>SL</label>
            <input type="number" value={stopLoss} onChange={e => setStopLoss(e.target.value)} className="input-small" />
            <label>TP</label>
            <input type="number" value={takeProfit} onChange={e => setTakeProfit(e.target.value)} className="input-small" />
          </div>
        </div>

        <div className="control-group">
             <span className="control-label">Opcje</span>
             <label className="trailing-label">
                <input type="checkbox" checked={trailingStop} onChange={e => setTrailingStop(e.target.checked)} />
                Trailing
             </label>
        </div>

        <button onClick={runSimulation} disabled={loading} className="btn-start">
            {loading ? '...' : '▶ Start'}
        </button>
      </div>

      {/* WYNIKI */}
      {results && (
        <div className="results-container">
            
            <div className="stats-grid">
                <div 
                  className="stat-box" 
                  style={{borderColor: results.total_return > results.buy_and_hold_return ? '#00e676' : '#444'}}
                >
                    <div className="stat-label">Strategia</div>
                    <div className={`stat-val ${results.total_return >= 0 ? 'green' : 'red'}`}>
                        {results.total_return > 0 ? '+' : ''}{results.total_return}%
                    </div>
                    <div className="stat-sub">{results.final_value.toLocaleString()} $</div>
                </div>
                
                <div className="stat-box">
                    <div className="stat-label">Rynek</div>
                    <div className="stat-val stat-val-blue">
                        {results.buy_and_hold_return > 0 ? '+' : ''}{results.buy_and_hold_return}%
                    </div>
                </div>

                <div className="stat-box">
                    <div className="stat-label">Sharpe</div>
                    <div className={`stat-val ${results.sharpe_ratio >= 1.0 ? 'green' : (results.sharpe_ratio < 0.5 ? 'red' : '')}`}>
                        {results.sharpe_ratio}
                    </div>
                </div>

                <div className="stat-box">
                    <div className="stat-label">Max Drawdown</div>
                    <div className="stat-val red">-{results.max_drawdown}%</div>
                </div>
            </div>

            {/* WYKRES Z MARKERAMI */}
            <div className="backtest-chart">
                 <Plot
                    data={plotData}
                    layout={{
                        title: { text: `Symulacja: ${results.trades_count} transakcji`, font: { size: 14, color: '#888' } },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: '#d1d4dc' },
                        margin: { t: 40, b: 40, l: 60, r: 20 },
                        xaxis: { showgrid: false },
                        yaxis: { title: 'Wartość Portfela ($)', gridcolor: '#2a2e39' },
                        legend: { orientation: 'h', y: 1.05, x: 0.5, xanchor: 'center' },
                        hovermode: 'closest'
                    }}
                    useResizeHandler={true}
                    className="plot-full-size"
                 />
            </div>

            {/* TABELA */}
            <div className="trades-table-container">
                <table className="trades-table">
                    <thead>
                        <tr className="trades-table-header">
                            <th>Data</th>
                            <th>Typ</th>
                            <th>Cena</th>
                            <th>Powód</th>
                            <th>Zysk</th>
                        </tr>
                    </thead>
                    <tbody>
                        {results.trades.slice().reverse().map((t, i) => (
                            <tr key={i} className="trades-table-row">
                                <td>{t.date}</td>
                                <td className={t.type === 'BUY' ? 'trade-type-buy' : 'trade-type-sell'}>{t.type}</td>
                                <td>{t.price.toFixed(2)}</td>
                                <td className="trade-reason">{t.reason}</td>
                                <td>
                                    {t.profit_pct !== undefined ? (
                                        <span className={t.profit_pct > 0 ? 'profit-positive' : 'profit-negative'}>
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
