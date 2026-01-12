import React, { useState } from 'react';
import Plot from 'react-plotly.js';

const NeuroPanel = ({ ticker, features }) => {
  const [trainingData, setTrainingData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [epochs, setEpochs] = useState(200);
  
  const [progress, setProgress] = useState(0);
  const [currentLoss, setCurrentLoss] = useState(null);

  const startTraining = async () => {
    if (!features || Object.keys(features).length === 0) {
      alert("Najpierw wygeneruj strategię (cechy)!");
      return;
    }

    setLoading(true);
    setTrainingData(null);
    setProgress(0);
    setCurrentLoss(null);

    try {
      const response = await fetch('http://127.0.0.1:5000/api/train_neuro', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker,
          features: Object.keys(features),
          epochs: parseInt(epochs)
        })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); 

        for (const line of lines) {
            if (!line.trim()) continue;
            try {
                const msg = JSON.parse(line);
                if (msg.status === 'progress') {
                    setProgress(msg.percent);
                    setCurrentLoss(msg.loss);
                } else if (msg.status === 'done') {
                    setProgress(100);
                    setTrainingData(msg.data);
                } else if (msg.error) {
                    alert("Błąd: " + msg.error);
                }
            } catch (e) { console.error(e); }
        }
      }
    } catch (err) {
      console.error(err);
      alert("Błąd połączenia.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '10px', background: '#1e222d', borderRadius: '8px', border: '1px solid #363a45', marginTop: '20px' }}>
      
      <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'15px', flexWrap:'wrap', gap:'10px'}}>
        <h4 style={{margin:0, color: '#cba6f7'}}>🧠 Neuro-Fuzzy Lab</h4>
        <div style={{display:'flex', gap:'10px', alignItems:'center'}}>
            <span style={{fontSize:'0.7em', color:'#888'}}>Epoki:</span>
            <input 
                type="number" value={epochs} onChange={(e) => setEpochs(e.target.value)}
                min="10" max="5000" step="50"
                style={{background: '#131722', color: '#fff', border: '1px solid #444', padding: '5px', width: '60px', textAlign: 'right'}}
            />
            <button 
              onClick={startTraining} disabled={loading} className="ai-button"
              style={{background: 'linear-gradient(135deg, #7c4dff 0%, #448aff 100%)', height: '38px', minWidth: '100px'}}
            >
              {loading ? `${progress}%` : '▶ Start AI'}
            </button>
        </div>
      </div>

      {loading && (
          <div style={{width: '100%', height: '24px', background: '#333', borderRadius: '12px', overflow: 'hidden', marginBottom: '20px', position: 'relative'}}>
              <div style={{width: `${progress}%`, height: '100%', background: 'linear-gradient(90deg, #7c4dff, #00e676)', transition: 'width 0.2s ease-out'}}></div>
              <div style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8em', color: '#fff', fontWeight: 'bold', textShadow: '0 1px 2px black'}}>
                  Epoka: {Math.round((progress / 100) * epochs)} / {epochs}
              </div>
          </div>
      )}

      {trainingData && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
            {/* WYKRES 1: BŁĄD */}
            <div style={{ height: '350px', background: '#131722', padding:'10px', borderRadius:'5px' }}>
                <Plot
                    data={[{
                        y: trainingData.loss_history,
                        type: 'scatter', mode: 'lines',
                        name: 'Błąd (MSE)',
                        line: { color: '#ff5252', width: 2 },
                        hovertemplate: 'Epoka: %{x}<br>Błąd: %{y:.5f}<extra></extra>'
                    }]}
                    layout={{
                        title: {text: 'Postęp Uczenia (Spadek Błędu)', font:{size:12, color:'#ccc'}},
                        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: '#aaa' },
                        margin: { t: 30, b: 40, l: 50, r: 20 },
                        xaxis: { title: 'Epoki', gridcolor: '#2a2e39' }, yaxis: { title: 'Błąd', gridcolor: '#2a2e39' }
                    }}
                    useResizeHandler={true} style={{width:'100%', height:'100%'}}
                />
            </div>

            {/* WYKRES 2: HISTORIA VS PROGNOZA (BOGATY) */}
            <div style={{ height: '350px', background: '#131722', padding:'10px', borderRadius:'5px' }}>
                 <Plot
                    data={[
                        {
                            // HISTORIA REALNA
                            x: trainingData.dates_history.slice(-100),
                            y: trainingData.actual.slice(-100),
                            customdata: trainingData.prices_history.slice(-100), // Przekazujemy CENĘ do tooltipa
                            type: 'scatter', mode: 'lines',
                            name: 'Realne',
                            line: { color: 'rgba(255,255,255,0.2)', width: 1 },
                            hovertemplate: '<b>%{x}</b><br>Zwrot: %{y:.2f}%<br>Cena: %{customdata:.2f}$<extra></extra>'
                        },
                        {
                            // HISTORIA NAUCZONA
                            x: trainingData.dates_history.slice(-100),
                            y: trainingData.predictions.slice(-100),
                            customdata: trainingData.prices_history.slice(-100),
                            type: 'scatter', mode: 'lines',
                            name: 'Model AI',
                            line: { color: '#00e676', width: 2 },
                            hovertemplate: '<b>%{x}</b><br>AI Zwrot: %{y:.2f}%<br>Cena: %{customdata:.2f}$<extra></extra>'
                        },
                        {
                            // PROGNOZA PRZYSZŁA
                            x: trainingData.dates_future,
                            y: trainingData.future_predictions,
                            customdata: trainingData.prices_future_implied, // Przekazujemy WYLICZONĄ cenę
                            type: 'scatter', mode: 'lines+markers',
                            name: 'PROGNOZA',
                            line: { color: '#ff9100', width: 2, dash: 'dot' },
                            marker: { size: 6 },
                            hovertemplate: '<b>%{x}</b> (Przyszłość)<br>Prognoza: %{y:.2f}%<br>Est. Cena: %{customdata:.2f}$<extra></extra>'
                        }
                    ]}
                    layout={{
                        title: {text: 'Prognoza Zwrotu (+ Cena w dymku)', font:{size:12, color:'#ccc'}},
                        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: '#aaa' },
                        margin: { t: 30, b: 60, l: 50, r: 20 },
                        showlegend: true, 
                        legend: { orientation: 'h', y: 1.1, font: {size: 10} },
                        xaxis: { 
                            title: 'Data', 
                            gridcolor: '#2a2e39',
                            tickangle: -45 // Pochylone daty dla czytelności
                        },
                        yaxis: { 
                            title: 'Zmiana (%)', 
                            gridcolor: '#2a2e39',
                            zeroline: true, zerolinecolor: '#444'
                        }
                    }}
                    useResizeHandler={true} style={{width:'100%', height:'100%'}}
                />
            </div>
        </div>
      )}
    </div>
  );
};

export default NeuroPanel;