// frontend/src/MarketChart.jsx
import React from 'react';
import Plot from 'react-plotly.js';

const MarketChart = ({ data }) => {
  if (!data) return <div style={{padding: 20}}>Brak danych do wyświetlenia.</div>;

  // Główny wykres ceny
  const tracePrice = {
    x: data.Date, y: data.Price,
    name: 'Cena Aktywa', type: 'scatter', mode: 'lines',
    line: { color: '#2962ff', width: 2 }, 
    xaxis: 'x', yaxis: 'y'
  };

  // Główny wynik ANFIS
  const traceAnfis = {
    x: data.Date, y: data.Sentiment_Oscillator,
    name: 'ANFIS (Wynik)', type: 'scatter', mode: 'lines',
    line: { color: '#f38ba8', width: 3 }, 
    fill: 'tozeroy', fillcolor: 'rgba(243, 139, 168, 0.1)',
    xaxis: 'x', yaxis: 'y2'
  };

  // --- DODATKOWE WSKAŹNIKI (Domyślnie ukryte w legendzie) ---
  // Wszystkie mapujemy na 'y2' (oś 0-100), żeby można je było porównać z ANFISem
  
  const traceRSI = {
    x: data.Date, y: data.RSI,
    name: 'RSI', type: 'scatter', mode: 'lines',
    line: { color: '#cba6f7', width: 1, dash: 'dot' },
    visible: 'legendonly', // Domyślnie wyłączony
    xaxis: 'x', yaxis: 'y2'
  };

  const traceVIX = {
    x: data.Date, y: data.VIX_Rank, // Używamy znormalizowanego Rangu 0-100
    name: 'VIX (Rank)', type: 'scatter', mode: 'lines',
    line: { color: '#fab387', width: 1, dash: 'dot' },
    visible: 'legendonly',
    xaxis: 'x', yaxis: 'y2'
  };

  const traceYield = {
    x: data.Date, y: data.Yield_Rank,
    name: 'Yield Curve (Rank)', type: 'scatter', mode: 'lines',
    line: { color: '#a6e3a1', width: 1, dash: 'dot' },
    visible: 'legendonly',
    xaxis: 'x', yaxis: 'y2'
  };

  const traceMACD = {
    x: data.Date, y: data.MACD_Rank,
    name: 'MACD (Rank)', type: 'scatter', mode: 'lines',
    line: { color: '#89b4fa', width: 1, dash: 'dot' },
    visible: 'legendonly',
    xaxis: 'x', yaxis: 'y2'
  };

  const traceM2 = {
    x: data.Date, y: data.M2_Rank,
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
      orientation: 'h', // Pozioma legenda
      y: 1.05, // Nad wykresem
      x: 0.5,
      xanchor: 'center'
    },

    shapes: [
        { type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y2', y0: 20, y1: 20, line: { color: 'green', width: 1, dash: 'dot', opacity: 0.5 } },
        { type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y2', y0: 80, y1: 80, line: { color: 'red', width: 1, dash: 'dot', opacity: 0.5 } }
    ]
  };

  return (
    <Plot
      data={[tracePrice, traceAnfis, traceRSI, traceVIX, traceYield, traceMACD, traceM2]}
      layout={layout}
      useResizeHandler={true}
      style={{ width: "100%", height: "100%" }}
      config={{ responsive: true, displayModeBar: true }}
    />
  );
};

export default MarketChart;