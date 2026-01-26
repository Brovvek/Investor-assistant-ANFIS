import React from 'react';
import Plot from 'react-plotly.js';
import './App.css';

const MarketChart = ({ data }) => {
  if (!data) return <div className="no-data-message">Brak danych do wyświetlenia.</div>;

  const tracePrice = {
    x: data.Date, y: data.Price,
    name: 'Cena Aktywa', type: 'scatter', mode: 'lines',
    line: { color: '#2962ff', width: 2 }, 
    xaxis: 'x', yaxis: 'y'
  };

  const traceAnfis = {
    x: data.Date, y: data.Sentiment_Oscillator,
    name: 'FIS (Wynik)', type: 'scatter', mode: 'lines',
    line: { color: '#f38ba8', width: 3 }, 
    fill: 'tozeroy', fillcolor: 'rgba(243, 139, 168, 0.1)',
    xaxis: 'x', yaxis: 'y2'
  };

  // --- WSKAŹNIKI POMOCNICZE (Oś Y2 0-100) ---
  // Uwaga: Backend teraz zwraca znormalizowane rangi (0-100) w tych polach.
  // Dzięki temu wszystko pasuje do skali ANFIS.
  
  const traceRSI = {
    x: data.Date, y: data.RSI,
    name: 'RSI (Rank)', type: 'scatter', mode: 'lines',
    line: { color: '#cba6f7', width: 1, dash: 'dot' },
    visible: 'legendonly',
    xaxis: 'x', yaxis: 'y2'
  };

  const traceVIX = {
    x: data.Date, y: data.VIX,
    name: 'VIX (Rank)', type: 'scatter', mode: 'lines',
    line: { color: '#fab387', width: 1, dash: 'dot' },
    visible: 'legendonly',
    xaxis: 'x', yaxis: 'y2'
  };

  const traceYield = {
    x: data.Date, y: data.Yield_Curve,
    name: 'Yield Curve (Rank)', type: 'scatter', mode: 'lines',
    line: { color: '#a6e3a1', width: 1, dash: 'dot' },
    visible: 'legendonly',
    xaxis: 'x', yaxis: 'y2'
  };

  const traceMACD = {
    x: data.Date, y: data.MACD,
    name: 'MACD Hist (Rank)', type: 'scatter', mode: 'lines',
    line: { color: '#89b4fa', width: 1, dash: 'dot' },
    visible: 'legendonly',
    xaxis: 'x', yaxis: 'y2'
  };

  const traceM2 = {
    x: data.Date, y: data.M2_Liquidity,
    name: 'M2 Płynność (Rank)', type: 'scatter', mode: 'lines',
    line: { color: '#f9e2af', width: 1, dash: 'dot' },
    visible: 'legendonly',
    xaxis: 'x', yaxis: 'y2'
  };

  const layout = {
    grid: { rows: 2, columns: 1, pattern: 'independent', roworder: 'top to bottom' },
    paper_bgcolor: '#131722', 
    plot_bgcolor: '#1e222d', 
    font: { color: '#d1d4dc' },
    margin: { t: 30, b: 30, l: 60, r: 50 }, 
    hovermode: 'x unified',
    
    // Oś X (Wspólna)
    xaxis: { anchor: 'y2', showgrid: true, gridcolor: '#2a2e39' },

    // Oś Y1 (Cena - Górne 60%)
    yaxis: { 
      domain: [0.45, 1], 
      title: 'Cena', 
      gridcolor: '#2a2e39' 
    },

    // Oś Y2 (Oscylatory 0-100 - Dolne 35%)
    yaxis2: { 
      domain: [0, 0.35], 
      title: 'Sentyment (0-100)', 
      range: [0, 100], 
      gridcolor: '#2a2e39' 
    },
    
    legend: {
      orientation: 'h',
      y: 1.05,
      x: 0.5,
      xanchor: 'center'
    },

    shapes: [
        { type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y2', y0: 20, y1: 20, line: { color: 'green', width: 1, dash: 'dot', opacity: 0.5 } },
        { type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y2', y0: 50, y1: 50, line: { color: 'gray', width: 1, dash: 'dot', opacity: 0.3 } },
        { type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y2', y0: 80, y1: 80, line: { color: 'red', width: 1, dash: 'dot', opacity: 0.5 } }
    ]
  };

  return (
    <Plot
      data={[tracePrice, traceAnfis, traceRSI, traceVIX, traceYield, traceMACD, traceM2]}
      layout={layout}
      useResizeHandler={true}
      className="plot-full-size"
      config={{ responsive: true, displayModeBar: true }}
    />
  );
};

export default MarketChart;
