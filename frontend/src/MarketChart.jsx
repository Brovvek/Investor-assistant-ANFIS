
// src/MarketChart.jsx
import React from 'react';
import Plot from 'react-plotly.js';

const MarketChart = ({ data }) => {
  if (!data) return <div style={{padding: 20}}>Brak danych do wyświetlenia.</div>;

  // Konfiguracja serii danych (Traces)
  const traces = [
    // Panel 1: Cena
    {
      x: data.Date, y: data.Price,
      name: 'Cena', type: 'scatter', mode: 'lines',
      line: { color: '#2962ff', width: 2 },
      xaxis: 'x', yaxis: 'y'
    },
    // Panel 2: ANFIS
    {
      x: data.Date, y: data.Sentiment_Oscillator,
      name: 'ANFIS', type: 'scatter', mode: 'lines',
      line: { color: '#f38ba8', width: 2 },
      fill: 'tozeroy', fillcolor: 'rgba(243, 139, 168, 0.1)',
      xaxis: 'x', yaxis: 'y2'
    },
    // Panel 3: RSI
    {
      x: data.Date, y: data.RSI,
      name: 'RSI', type: 'scatter', mode: 'lines',
      line: { color: '#cba6f7', width: 1.5 },
      xaxis: 'x', yaxis: 'y3'
    },
    // Panel 4: VIX
    {
      x: data.Date, y: data.VIX,
      name: 'VIX', type: 'scatter', mode: 'lines',
      line: { color: '#fab387', width: 1.5 },
      xaxis: 'x', yaxis: 'y4'
    },
    // Panel 4: Yield (Prawa oś)
    {
      x: data.Date, y: data.Yield_Curve,
      name: 'Yield Spread', type: 'scatter', mode: 'lines',
      line: { color: '#a6e3a1', width: 1.5, dash: 'dot' },
      xaxis: 'x', yaxis: 'y5'
    }
  ];

  // Układ (Layout)
  const layout = {
    grid: { rows: 4, columns: 1, pattern: 'independent', roworder: 'top to bottom' },
    paper_bgcolor: '#131722',
    plot_bgcolor: '#1e222d',
    font: { color: '#d1d4dc' },
    margin: { t: 30, b: 30, l: 60, r: 50 },
    hovermode: 'x unified',
    autosize: true,
    xaxis: { anchor: 'y4', showgrid: true, gridcolor: '#2a2e39' },
    yaxis: { domain: [0.62, 1], title: 'Cena', gridcolor: '#2a2e39' },
    yaxis2: { domain: [0.42, 0.60], title: 'ANFIS', range: [0, 100], gridcolor: '#2a2e39' },
    yaxis3: { domain: [0.22, 0.40], title: 'RSI', range: [0, 100], gridcolor: '#2a2e39', tickvals: [30, 70] },
    yaxis4: { domain: [0, 0.20], title: 'VIX', titlefont: {color: '#fab387'}, gridcolor: '#2a2e39' },
    yaxis5: { domain: [0, 0.20], title: 'Yield', titlefont: {color: '#a6e3a1'}, overlaying: 'y4', side: 'right', showgrid: false },
    showlegend: true,
    legend: { orientation: 'h', y: 1.05 },
    shapes: [
        { type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y2', y0: 20, y1: 20, line: { color: 'green', width: 1, dash: 'dot' } },
        { type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y2', y0: 80, y1: 80, line: { color: 'red', width: 1, dash: 'dot' } },
        { type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y3', y0: 30, y1: 30, line: { color: 'white', width: 0.5, dash: 'dot' } },
        { type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y3', y0: 70, y1: 70, line: { color: 'white', width: 0.5, dash: 'dot' } },
        { type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y5', y0: 0, y1: 0, line: { color: 'red', width: 1 } },
    ]
  };

  return (
    <Plot
      data={traces}
      layout={layout}
      useResizeHandler={true}
      style={{ width: "100%", height: "100%" }}
      config={{ responsive: true, displayModeBar: true }}
    />
  );
};

export default MarketChart;