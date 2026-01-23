import React, { useState } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './App.css';

const BacktestPanel = ({ ticker, config }) => {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const [buyThresh, setBuyThresh] = useState(65);
  const [sellThresh, setSellThresh] = useState(35);
  const [stopLoss, setStopLoss] = useState(5);
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
        alert("BÅ‚Ä…d symulacji. SprawdÅº czy pobrano dane w sekcji Analizy."); 
    } finally { 
        setLoading(false); 
    }
  };

  // --- PRZYGOTOWANIE DANYCH DO WYKRESU ---
  let plotData = [];
  
  if (results && results.equity_curve && results.chart_dates) {
      
      // 1. Mapa pomocnicza: Data -> WartoÅ›Ä‡ Portfela (Equity)
      // DziÄ™ki temu wiemy, na jakiej wysokoÅ›ci narysowaÄ‡ trÃ³jkÄ…t w danym dniu
      const dateToEquity = {};
      results.chart_dates.forEach((date, index) => {
          dateToEquity[date] = results.equity_curve[index];
      });

      // 2. Przygotowanie markerÃ³w KUPNA i SPRZEDAÅ»Y
      const buys = { x: [], y: [], text: [] };
      const sells = { x: [], y: [], text: [] };

      results.trades.forEach(trade => {
          const equityValue = dateToEquity[trade.date];
          // JeÅ›li z jakiegoÅ› powodu daty nie ma w mapie (rzadkie), pomijamy
          if (equityValue === undefined) return;

          const tooltip = `Cena: ${trade.price.toFixed(2)}<br>PowÃ³d: ${trade.reason}`;

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
        fillcolor: 'rgba(0, 230, 118, 0.05)' // Bardzo delikatne tÅ‚o
      });

      // Markery KUPNA (Zielone trÃ³jkÄ…ty w gÃ³rÄ™)
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

      // Markery SPRZEDAÅ»Y (Czerwone trÃ³jkÄ…ty w dÃ³Å‚)
      if (sells.x.length > 0) {
          plotData.push({
              x: sells.x,
              y: sells.y,
              mode: 'markers',
              name: 'SprzedaÅ¼',
              type: 'scatter',
              marker: { symbol: 'triangle-down', color: '#ff5252', size: 10, line: { color: 'white', width: 1 } },
              text: sells.text,
              hoverinfo: 'text+x'
          });
      }
  }

  return (
    <div style={{ padding: '10px' }}>
      
      {/* KONTROLKI STRATEGII */}
      <div className="controls-row" style={{ flexWrap: 'wrap', gap: '20px', alignItems: 'flex-end', borderTop: 'none', paddingTop: 0, justifyContent: 'center' }}>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
          <span style={{fontSize: '0.8em', color: '#888'}}>SygnaÅ‚y ANFIS</span>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <label>Kup &ge;</label>
            <input type="number" value={buyThresh} onChange={e => setBuyThresh(e.target.value)} style={{width: '50px'}} />
            <label>Sprzedaj &le;</label>
            <input type="number" value={sellThresh} onChange={e => setSellThresh(e.target.value)} style={{width: '50px'}} />
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
          <span style={{fontSize: '0.8em', color: '#888'}}>Ryzyko (%)</span>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <label>SL</label>
            <input type="number" value={stopLoss} onChange={e => setStopLoss(e.target.value)} style={{width: '50px'}} />
            <label>TP</label>
            <input type="number" value={takeProfit} onChange={e => setTakeProfit(e.target.value)} style={{width: '50px'}} />
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
             <span style={{fontSize: '0.8em', color: '#888'}}>Opcje</span>
             <label style={{display: 'flex', alignItems: 'center', gap: '5px', cursor: 'pointer', height: '30px'}}>
                <input type="checkbox" checked={trailingStop} onChange={e => setTrailingStop(e.target.checked)} />
                Trailing
             </label>
        </div>

        <button onClick={runSimulation} disabled={loading} style={{ height: '35px', padding: '0 20px' }}>
            {loading ? '...' : 'â–¶ Start'}
        </button>
      </div>

      {/* WYNIKI */}
      {results && (
        <div style={{ marginTop: '30px' }}>
            
            <div className="stats-grid">
                <div className="stat-box" style={{borderColor: results.total_return > results.buy_and_hold_return ? '#00e676' : '#444'}}>
                    <div className="stat-label">Strategia</div>
                    <div className={`stat-val ${results.total_return >= 0 ? 'green' : 'red'}`}>
                        {results.total_return > 0 ? '+' : ''}{results.total_return}%
                    </div>
                    <div className="stat-sub">{results.final_value.toLocaleString()} $</div>
                </div>
                
                <div className="stat-box">
                    <div className="stat-label">Rynek</div>
                    <div className="stat-val" style={{color: '#89b4fa'}}>
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
            <div style={{ marginTop: '20px', height: '450px', border: '1px solid #363a45', borderRadius: '8px', padding: '10px', background: '#1e222d' }}>
                 <Plot
                    data={plotData}
                    layout={{
                        title: { text: `Symulacja: ${results.trades_count} transakcji`, font: { size: 14, color: '#888' } },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: '#d1d4dc' },
                        margin: { t: 40, b: 40, l: 60, r: 20 },
                        xaxis: { showgrid: false },
                        yaxis: { title: 'WartoÅ›Ä‡ Portfela ($)', gridcolor: '#2a2e39' },
                        legend: { orientation: 'h', y: 1.05, x: 0.5, xanchor: 'center' },
                        hovermode: 'closest'
                    }}
                    useResizeHandler={true}
                    style={{ width: "100%", height: "100%" }}
                 />
            </div>

            {/* TABELA */}
            <div style={{ marginTop: '20px', maxHeight: '300px', overflowY: 'auto', background: '#1a1d26', padding: '15px', borderRadius: '8px', fontSize: '0.85em' }}>
                <table style={{width: '100%', borderCollapse: 'collapse', textAlign: 'left'}}>
                    <thead style={{position: 'sticky', top: 0, background: '#1a1d26'}}>
                        <tr style={{color: '#888'}}>
                            <th style={{padding:'5px'}}>Data</th>
                            <th>Typ</th>
                            <th>Cena</th>
                            <th>PowÃ³d</th>
                            <th>Zysk</th>
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