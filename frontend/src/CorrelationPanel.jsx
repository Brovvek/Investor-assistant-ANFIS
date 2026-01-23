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
      alert("BÅ‚Ä…d analizy korelacji");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (ticker) fetchCorrelations();
  }, [ticker]);

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '40px', color: '#888', background: '#1e222d', borderRadius: '8px' }}>
        <div className="loading-spinner"></div>
        <p style={{ marginTop: '15px' }}>â³ InÅ¼ynieria Cech w toku...</p>
        <p style={{ fontSize: '0.8em' }}>Generowanie setek wskaÅºnikÃ³w i badanie zaleÅ¼noÅ›ci...</p>
      </div>
    );
  }

  if (!corrData) return null;

  // --- PRZYGOTOWANIE DANYCH DLA PLOTLY ---
  
  // OÅ› X: Nazwy WskaÅºnikÃ³w - OGRANICZAMY DO TOP 20
  // Backend zwraca 50, ale my wyÅ›wietlamy tylko 20 najlepszych dla czytelnoÅ›ci
  const xValues = Object.keys(corrData).slice(0, 20);
  
  // OÅ› Y: Metody badawcze
  const yValues = ['Pearson (Liniowa)', 'Spearman (Rangowa)', 'Kendall (ZgodnoÅ›Ä‡)'];

  // OÅ› Z: WartoÅ›ci korelacji (Macierz)
  // Mapujemy tylko te xValues, ktÃ³re zostaÅ‚y po przyciÄ™ciu (Top 20)
  const zValues = [
    xValues.map(feat => corrData[feat]['pearson']),
    xValues.map(feat => corrData[feat]['spearman']),
    xValues.map(feat => corrData[feat]['kendall'])
  ];

  // Tekst do wyÅ›wietlenia w komÃ³rkach (Formatowanie %)
  const textValues = zValues.map(row => 
    row.map(val => `${(val * 100).toFixed(0)}%`)
  );

  return (
    <div style={{ width: '100%', padding: '10px', background: '#1e222d', borderRadius: '8px', border: '1px solid #363a45' }}>
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px'}}>
             <h4 style={{margin:0, color: '#d1d4dc'}}>
               ðŸ“Š Mapa Korelacji (Top 20 WskaÅºnikÃ³w)
               <span style={{fontSize: '0.7em', color: '#888', marginLeft: '10px', fontWeight: 'normal'}}>
                 (Cieplej = Silniejsza korelacja)
               </span>
             </h4>
             <button 
                onClick={fetchCorrelations} 
                className="ai-button"
                style={{fontSize: '0.8em', padding: '6px 12px', background: '#444', border: 'none'}}
             >
                OdÅ›wieÅ¼
             </button>
        </div>

        <div style={{ width: '100%', height: '350px' }}>
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
                        tickfont: { size: 11 } // Nieco wiÄ™ksza czcionka, bo jest mniej kolumn
                    },
                    yaxis: {
                        tickfont: { size: 12, style: 'bold' }
                    }
                }}
                useResizeHandler={true}
                style={{ width: "100%", height: "100%" }}
                config={{ displayModeBar: false }}
            />
        </div>
    </div>
  );
};

export default CorrelationPanel;