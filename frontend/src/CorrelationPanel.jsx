import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './App.css';

const CorrelationPanel = ({ ticker }) => {
  const [corrData, setCorrData] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchCorrelations = async () => {
    setLoading(true);
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/correlations', {
        ticker: ticker
      });
      setCorrData(response.data);
    } catch (err) {
      console.error(err);
      alert("Błąd analizy korelacji");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (ticker) fetchCorrelations();
  }, [ticker]);

  if (loading) {
    return (
      <div className="correlation-loading">
        <div className="loading-spinner"></div>
        <p className="correlation-loading-text">⏳ Inżynieria Cech w toku...</p>
        <p className="correlation-loading-subtext">Generowanie setek wskaźników i badanie zależności...</p>
      </div>
    );
  }

  if (!corrData) return null;

  // --- PRZYGOTOWANIE DANYCH DLA PLOTLY ---
  
  // Oś X: Nazwy Wskaźników - OGRANICZAMY DO TOP 20
  // Backend zwraca 50, ale my wyświetlamy tylko 20 najlepszych dla czytelności
  const xValues = Object.keys(corrData).slice(0, 20);
  
  // Oś Y: Metody badawcze
  const yValues = ['Pearson (Liniowa)', 'Spearman (Rangowa)', 'Kendall (Zgodność)'];

  // Oś Z: Wartości korelacji (Macierz)
  // Mapujemy tylko te xValues, które zostały po przycięciu (Top 20)
  const zValues = [
    xValues.map(feat => corrData[feat]['pearson']),
    xValues.map(feat => corrData[feat]['spearman']),
    xValues.map(feat => corrData[feat]['kendall'])
  ];

  // Tekst do wyświetlenia w komórkach (Formatowanie do 2 miejsc po przecinku)
  const textValues = zValues.map(row => 
    row.map(val => val.toFixed(2))

  );

  return (
    <div className="correlation-panel">
        <div className="correlation-header">
             <h4 className="correlation-title">
               📊 Mapa Korelacji (Top 20 Wskaźników)
               <span className="correlation-subtitle">
                 (Cieplej = Silniejsza korelacja)
               </span>
             </h4>
             <button 
                onClick={fetchCorrelations} 
                className="ai-button btn-refresh"
             >
                Odśwież
             </button>
        </div>

        <div className="correlation-chart">
            <Plot
                data={[{
                    x: xValues,
                    y: yValues,
                    z: zValues,
                    text: textValues,
                    texttemplate: "%{text}",
                    type: 'heatmap',
                    hoverongaps: false,
                    colorscale: [
                        [0, '#ef5350'],   // Czerwony (Ujemna)
                        [0.5, '#1e222d'], // Ciemny (Zero)
                        [1, '#00e676']    // Zielony (Dodatnia)
                    ],
                    zmin: -1,
                    zmax: 1,
                    xgap: 1,
                    ygap: 1
                }]}
                layout={{
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#d1d4dc' },
                    margin: { t: 30, b: 80, l: 120, r: 20 },
                    xaxis: { 
                        side: 'bottom',
                        tickangle: -45,
                        tickfont: { size: 11 }
                    },
                    yaxis: {
                        tickfont: { size: 12, style: 'bold' }
                    }
                }}
                useResizeHandler={true}
                className="plot-full-size"
                config={{ displayModeBar: false }}
            />
        </div>
    </div>
  );
};

export default CorrelationPanel;
