import React, { useState } from 'react';
import Plot from 'react-plotly.js';
import './App.css';

// ========== BEZPIECZNE FUNKCJE POMOCNICZE ==========

// Bezpieczne formatowanie liczby
const fmt = (value, decimals = 2) => {
  if (value === undefined || value === null || isNaN(value)) return '—';
  return Number(value).toFixed(decimals);
};

// ========== KOMPONENT ==========

const AnfisMLPanel = ({ ticker, config }) => {
  // Parametry treningu
  const [epochs, setEpochs] = useState(100);
  const [numMfs, setNumMfs] = useState(3);
  const [batchSize, setBatchSize] = useState(64);
  const [learningRate, setLearningRate] = useState(0.01);
  const [mfType, setMfType] = useState('gauss');
  const [hybrid, setHybrid] = useState(true);
  const [optimizer, setOptimizer] = useState('adam');
  const [lookahead, setLookahead] = useState(1);
  const [predictionType, setPredictionType] = useState('returns');
  const [scalerType, setScalerType] = useState('robust');
  const [earlyStoppingPatience, setEarlyStoppingPatience] = useState(20);
  const [trainingDays, setTrainingDays] = useState(0); // 0 = wszystkie dane
  
  // Stan treningu
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [liveMetrics, setLiveMetrics] = useState(null);
  
  // Wyniki
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [rawResponse, setRawResponse] = useState(''); // DEBUG
  
  // Aktywna zakładka
  const [activeTab, setActiveTab] = useState('predictions');

  // ========== FUNKCJA TRENINGU ==========
  
  const startTraining = async () => {
    setIsTraining(true);
    setProgress(0);
    setCurrentEpoch(0);
    setResults(null);
    setError('');
    setLiveMetrics(null);
    setRawResponse('');

    try {
      const response = await fetch('http://127.0.0.1:5000/api/anfis_ml/train_stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker,
          config,
          epochs: parseInt(epochs),
          num_mfs: parseInt(numMfs),
          batch_size: parseInt(batchSize),
          learning_rate: parseFloat(learningRate),
          mf_type: mfType,
          hybrid,
          optimizer,
          lookahead: parseInt(lookahead),
          prediction_type: predictionType,
          scaler_type: scalerType,
          early_stopping_patience: parseInt(earlyStoppingPatience),
          training_days: parseInt(trainingDays) || 0
        })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let lastData = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim()) {
            try {
              const data = JSON.parse(line);
              lastData = data;
              
              if (data.status === 'training') {
                setProgress(data.progress || 0);
                setCurrentEpoch(data.epoch || 0);
                setLiveMetrics({
                  val_rmse: data.val_rmse,
                  val_mape: data.val_mape,
                  direction_acc: data.direction_acc
                });
              } else if (data.status === 'done') {
                console.log('✅ Training complete:', data);
                setRawResponse(JSON.stringify(data, null, 2).slice(0, 2000));
                setResults(data);
                setProgress(100);
              } else if (data.status === 'error') {
                setError(data.message || 'Unknown error');
              }
            } catch (e) {
              console.error('JSON parse error:', e, 'Line:', line.slice(0, 200));
            }
          }
        }
      }
      
      // Jeśli nie otrzymano "done", ale mamy dane, użyj ich
      if (!results && lastData && lastData.status !== 'error') {
        console.log('Using last received data:', lastData);
        setResults(lastData);
      }
      
    } catch (err) {
      console.error('Training error:', err);
      setError(err.message || 'Błąd połączenia z serwerem');
    } finally {
      setIsTraining(false);
    }
  };

  // ========== SPRAWDZENIE KONFIGURACJI ==========
  
  const activeFeatures = Object.entries(config || {}).filter(([k, v]) => v && v.enabled);
  const canTrain = activeFeatures.length >= 1;

  // ========== POMOCNICZE GETTERY ==========
  
  const getMetric = (key, fallback = 0) => {
    if (!results || !results.metrics) return fallback;
    const m = results.metrics;
    if (key === 'rmse') return m.rmse ?? m.test_rmse ?? fallback;
    if (key === 'mape') return m.mape ?? m.test_mape ?? fallback;
    return m[key] ?? fallback;
  };

  const getPredictions = () => results?.predictions || { dates: [], actual: [], predicted: [] };
  const getHistory = () => results?.training_history || { epochs: [], train_loss: [], val_loss: [] };
  const getMFs = () => results?.membership_functions || {};
  const getRules = () => results?.rules || [];
  const getImportance = () => results?.feature_importance || {};

  // ========== RENDER ==========

  return (
    <div style={{ padding: '10px' }}>
      
      {/* INFO BOX */}
      <div style={{
        background: 'rgba(41, 98, 255, 0.1)',
        border: '1px solid #2962ff',
        borderRadius: '8px',
        padding: '12px',
        marginBottom: '15px',
        fontSize: '0.85em'
      }}>
        <strong>💡 Wskazówka:</strong> Wybierz <strong>"% Zmiana Ceny"</strong> jako typ predykcji.
        Direction Accuracy &gt; 55% = użyteczny model.
      </div>

      {/* PARAMETRY */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', 
        gap: '10px',
        marginBottom: '20px',
        padding: '15px',
        background: '#1a1d26',
        borderRadius: '8px',
        border: '1px solid #363a45'
      }}>
        <div>
          <label style={labelStyle}>Typ Predykcji ⭐</label>
          <select value={predictionType} onChange={e => setPredictionType(e.target.value)} style={inputStyle}>
            <option value="returns">% Zmiana Ceny</option>
            <option value="log_returns">Log Returns</option>
            <option value="direction">Kierunek (0/1)</option>
            <option value="price">Surowa Cena</option>
          </select>
        </div>
        
        <div>
          <label style={labelStyle}>Epoki</label>
          <input type="number" value={epochs} onChange={e => setEpochs(e.target.value)} min="10" max="500" style={inputStyle} />
        </div>
        
        <div>
          <label style={labelStyle}>MF Count</label>
          <input type="number" value={numMfs} onChange={e => setNumMfs(e.target.value)} min="2" max="7" style={inputStyle} />
        </div>
        
        <div>
          <label style={labelStyle}>MF Type</label>
          <select value={mfType} onChange={e => setMfType(e.target.value)} style={inputStyle}>
            <option value="gauss">Gaussian</option>
            <option value="bell">Bell</option>
            <option value="tri">Triangular</option>
          </select>
        </div>
        
        <div>
          <label style={labelStyle}>Optimizer</label>
          <select value={optimizer} onChange={e => setOptimizer(e.target.value)} style={inputStyle}>
            <option value="adam">Adam</option>
            <option value="adamw">AdamW</option>
            <option value="sgd">SGD</option>
          </select>
        </div>
        
        <div>
          <label style={labelStyle}>Learning Rate</label>
          <input type="number" value={learningRate} onChange={e => setLearningRate(e.target.value)} step="0.001" style={inputStyle} />
        </div>
        
        <div>
          <label style={labelStyle}>Batch Size</label>
          <select value={batchSize} onChange={e => setBatchSize(e.target.value)} style={inputStyle}>
            <option value="32">32</option>
            <option value="64">64</option>
            <option value="128">128</option>
          </select>
        </div>
        
        <div>
          <label style={labelStyle}>Lookahead (dni)</label>
          <input type="number" value={lookahead} onChange={e => setLookahead(e.target.value)} min="1" max="30" style={inputStyle} />
        </div>

        <div>
          <label style={labelStyle}>Scaler</label>
          <select value={scalerType} onChange={e => setScalerType(e.target.value)} style={inputStyle}>
            <option value="robust">Robust</option>
            <option value="standard">Standard</option>
            <option value="minmax">MinMax</option>
          </select>
        </div>

        <div>
          <label style={labelStyle}>Early Stop</label>
          <input type="number" value={earlyStoppingPatience} onChange={e => setEarlyStoppingPatience(e.target.value)} min="5" max="50" style={inputStyle} />
        </div>

        <div>
          <label style={labelStyle}>Dni treningowe</label>
          <select value={trainingDays} onChange={e => setTrainingDays(e.target.value)} style={inputStyle}>
            <option value="0">Wszystkie dane</option>
            <option value="252">1 rok (252)</option>
            <option value="504">2 lata (504)</option>
            <option value="756">3 lata (756)</option>
            <option value="1260">5 lat (1260)</option>
            <option value="2520">10 lat (2520)</option>
          </select>
        </div>
        
        <div style={{ display: 'flex', alignItems: 'center', paddingTop: '20px' }}>
          <label>
            <input type="checkbox" checked={hybrid} onChange={e => setHybrid(e.target.checked)} style={{ marginRight: '5px' }} />
            Hybrid (LSE)
          </label>
        </div>
      </div>

      {/* PRZYCISK STARTU */}
      <div style={{ display: 'flex', gap: '15px', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap' }}>
        <button 
          onClick={startTraining} 
          disabled={isTraining || !canTrain}
          style={{ 
            padding: '12px 30px', 
            fontSize: '1em',
            background: canTrain ? 'linear-gradient(135deg, #2962ff 0%, #00c853 100%)' : '#444',
            border: 'none',
            borderRadius: '8px',
            color: 'white',
            cursor: canTrain && !isTraining ? 'pointer' : 'not-allowed'
          }}
        >
          {isTraining ? `Trenuję... ${progress}%` : '🧠 Rozpocznij Uczenie ANFIS'}
        </button>
        
        {!canTrain && (
          <span style={{ color: '#ff5252', fontSize: '0.9em' }}>⚠️ Wybierz min. 1 wskaźnik</span>
        )}
        
        {isTraining && liveMetrics && (
          <div style={{ fontSize: '0.8em', color: '#888', background: 'rgba(41,98,255,0.1)', padding: '8px 12px', borderRadius: '4px' }}>
            Epoka: {currentEpoch}/{epochs} | RMSE: {fmt(liveMetrics.val_rmse, 4)} | Dir: {fmt(liveMetrics.direction_acc, 1)}%
          </div>
        )}
      </div>

      {/* PASEK POSTĘPU */}
      {isTraining && (
        <div style={{ height: '6px', background: '#2a2e39', borderRadius: '3px', marginBottom: '20px', overflow: 'hidden' }}>
          <div style={{ height: '100%', width: `${progress}%`, background: 'linear-gradient(90deg, #2962ff, #00c853)', transition: 'width 0.3s' }} />
        </div>
      )}

      {/* BŁĄD */}
      {error && (
        <div style={{ background: 'rgba(255,82,82,0.1)', border: '1px solid #ff5252', borderRadius: '8px', padding: '15px', marginBottom: '20px', color: '#ff5252' }}>
          ❌ {error}
        </div>
      )}

      {/* ========== WYNIKI ========== */}
      {results && (
        <div>
          {/* METRYKI - BEZPIECZNE */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '15px', marginBottom: '20px' }}>
            
            <div style={statBox(getMetric('direction_accuracy', 50) > 55)}>
              <div style={statLabel}>Direction Accuracy ⭐</div>
              <div style={{...statVal, color: getMetric('direction_accuracy', 50) > 55 ? '#00e676' : '#d1d4dc'}}>
                {fmt(getMetric('direction_accuracy', 50), 1)}%
              </div>
            </div>
            
            <div style={statBox()}>
              <div style={statLabel}>RMSE</div>
              <div style={{...statVal, color: '#89b4fa'}}>{fmt(getMetric('rmse'), 4)}</div>
            </div>
            
            <div style={statBox()}>
              <div style={statLabel}>MAPE</div>
              <div style={statVal}>{fmt(getMetric('mape'), 2)}%</div>
            </div>
            
            <div style={statBox(getMetric('r_squared', 0) > 0.2)}>
              <div style={statLabel}>R²</div>
              <div style={{...statVal, color: getMetric('r_squared', 0) > 0.2 ? '#00e676' : '#d1d4dc'}}>
                {fmt(getMetric('r_squared', 0) * 100, 1)}%
              </div>
            </div>
            
            <div style={statBox()}>
              <div style={statLabel}>Correlation</div>
              <div style={statVal}>{fmt(getMetric('correlation', 0) * 100, 1)}%</div>
            </div>
            
            <div style={statBox()}>
              <div style={statLabel}>Train / Test</div>
              <div style={{...statVal, color: '#cba6f7', fontSize: '1.2em'}}>
                {getMetric('train_samples', '?')} / {getMetric('test_samples', '?')}
              </div>
            </div>

            {getMetric('win_rate') > 0 && (
              <div style={statBox(getMetric('win_rate', 50) > 50)}>
                <div style={statLabel}>Win Rate</div>
                <div style={{...statVal, color: getMetric('win_rate', 50) > 50 ? '#00e676' : '#ff5252'}}>
                  {fmt(getMetric('win_rate'), 1)}%
                </div>
              </div>
            )}

            {getMetric('strategy_sharpe') !== 0 && (
              <div style={statBox(getMetric('strategy_sharpe', 0) > 1)}>
                <div style={statLabel}>Sharpe Ratio</div>
                <div style={{...statVal, color: getMetric('strategy_sharpe', 0) > 1 ? '#00e676' : '#d1d4dc'}}>
                  {fmt(getMetric('strategy_sharpe'), 2)}
                </div>
              </div>
            )}
          </div>

          {/* ZAKŁADKI */}
          <div style={{ display: 'flex', gap: '5px', marginBottom: '15px', borderBottom: '1px solid #363a45', paddingBottom: '10px', flexWrap: 'wrap' }}>
            {['predictions', 'scatter', 'loss', 'mf', 'importance', 'rules', 'debug'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                style={{
                  padding: '8px 12px',
                  background: activeTab === tab ? '#2962ff' : 'transparent',
                  border: activeTab === tab ? 'none' : '1px solid #363a45',
                  borderRadius: '4px',
                  color: activeTab === tab ? '#fff' : '#888',
                  cursor: 'pointer',
                  fontSize: '0.8em'
                }}
              >
                {tab === 'predictions' && '📈 Predykcje'}
                {tab === 'scatter' && '🎯 Scatter'}
                {tab === 'loss' && '📉 Loss'}
                {tab === 'mf' && '🔔 MF'}
                {tab === 'importance' && '⚖️ Ważność'}
                {tab === 'rules' && '📜 Reguły'}
                {tab === 'debug' && '🐛 Debug'}
              </button>
            ))}
          </div>

          {/* ZAWARTOŚĆ ZAKŁADEK */}
          <div style={{ background: '#1e222d', borderRadius: '8px', padding: '15px', border: '1px solid #363a45', minHeight: '350px' }}>
            
            {/* PREDYKCJE */}
            {activeTab === 'predictions' && (
              getPredictions().dates?.length > 0 ? (
                <Plot
                  data={[
                    { x: getPredictions().dates, y: getPredictions().actual, type: 'scatter', mode: 'lines', name: 'Actual', line: { color: '#2962ff', width: 2 } },
                    { x: getPredictions().dates, y: getPredictions().predicted, type: 'scatter', mode: 'lines', name: 'Predicted', line: { color: '#00e676', width: 2, dash: 'dot' } },
                    ...(predictionType !== 'price' ? [{ x: getPredictions().dates, y: getPredictions().dates.map(() => 0), type: 'scatter', mode: 'lines', line: { color: '#666', width: 1, dash: 'dash' }, showlegend: false }] : [])
                  ]}
                  layout={{
                    title: { text: `Predykcja: ${predictionType} (${lookahead}d)`, font: { color: '#d1d4dc', size: 14 } },
                    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                    font: { color: '#d1d4dc' },
                    margin: { t: 40, b: 40, l: 60, r: 20 },
                    xaxis: { showgrid: false },
                    yaxis: { gridcolor: '#2a2e39' },
                    legend: { orientation: 'h', y: 1.1 }
                  }}
                  useResizeHandler style={{ width: '100%', height: '380px' }}
                />
              ) : <div style={emptyState}>Brak danych predykcji</div>
            )}

            {/* SCATTER */}
            {activeTab === 'scatter' && (
              getPredictions().actual?.length > 0 ? (
                <Plot
                  data={[
                    {
                      x: getPredictions().actual, y: getPredictions().predicted,
                      type: 'scatter', mode: 'markers', name: 'Points',
                      marker: { color: getPredictions().actual.map((a, i) => Math.sign(a) === Math.sign(getPredictions().predicted[i]) ? '#00e676' : '#ff5252'), size: 5, opacity: 0.7 }
                    },
                    {
                      x: [Math.min(...getPredictions().actual), Math.max(...getPredictions().actual)],
                      y: [Math.min(...getPredictions().actual), Math.max(...getPredictions().actual)],
                      type: 'scatter', mode: 'lines', name: 'Perfect', line: { color: '#888', dash: 'dash' }
                    }
                  ]}
                  layout={{
                    title: { text: `Actual vs Predicted (R²=${fmt(getMetric('r_squared') * 100, 1)}%)`, font: { color: '#d1d4dc', size: 14 } },
                    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
                    font: { color: '#d1d4dc' },
                    margin: { t: 40, b: 50, l: 60, r: 20 },
                    xaxis: { title: 'Actual', gridcolor: '#2a2e39' },
                    yaxis: { title: 'Predicted', gridcolor: '#2a2e39' }
                  }}
                  useResizeHandler style={{ width: '100%', height: '380px' }}
                />
              ) : <div style={emptyState}>Brak danych</div>
            )}

            {/* LOSS */}
            {activeTab === 'loss' && (
              getHistory().epochs?.length > 0 ? (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  <Plot
                    data={[
                      { x: getHistory().epochs, y: getHistory().train_loss, type: 'scatter', mode: 'lines', name: 'Train', line: { color: '#2962ff' } },
                      { x: getHistory().epochs, y: getHistory().val_loss, type: 'scatter', mode: 'lines', name: 'Val', line: { color: '#ff5252' } }
                    ]}
                    layout={{ title: { text: 'Loss', font: { color: '#d1d4dc', size: 12 } }, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#d1d4dc', size: 9 }, margin: { t: 35, b: 35, l: 45, r: 10 }, xaxis: { showgrid: false }, yaxis: { gridcolor: '#2a2e39' }, legend: { orientation: 'h', y: 1.15 } }}
                    style={{ width: '100%', height: '250px' }}
                  />
                  <Plot
                    data={[
                      { x: getHistory().epochs, y: getHistory().val_direction_acc || getHistory().epochs.map(() => 50), type: 'scatter', mode: 'lines', name: 'DirAcc', line: { color: '#00e676' }, fill: 'tozeroy', fillcolor: 'rgba(0,230,118,0.1)' },
                      { x: getHistory().epochs, y: getHistory().epochs.map(() => 50), type: 'scatter', mode: 'lines', name: '50%', line: { color: '#666', dash: 'dash' } }
                    ]}
                    layout={{ title: { text: 'Direction Accuracy', font: { color: '#d1d4dc', size: 12 } }, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#d1d4dc', size: 9 }, margin: { t: 35, b: 35, l: 45, r: 10 }, xaxis: { showgrid: false }, yaxis: { gridcolor: '#2a2e39', range: [40, 70] }, legend: { orientation: 'h', y: 1.15 } }}
                    style={{ width: '100%', height: '250px' }}
                  />
                </div>
              ) : <div style={emptyState}>Brak historii treningu</div>
            )}

            {/* MF */}
            {activeTab === 'mf' && (
              Object.keys(getMFs()).length > 0 ? (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '10px' }}>
                  {Object.entries(getMFs()).map(([name, data]) => (
                    <div key={name} style={{ background: '#1a1d26', borderRadius: '6px', padding: '8px' }}>
                      <Plot
                        data={Object.entries(data.mfs || {}).map(([mfName, yVals], idx) => ({
                          x: data.x || [], y: yVals || [], type: 'scatter', mode: 'lines', name: mfName,
                          line: { color: ['#2962ff', '#00e676', '#ff5252', '#fab387', '#cba6f7'][idx % 5], width: 2 }
                        }))}
                        layout={{ title: { text: name, font: { color: '#d1d4dc', size: 11 } }, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#d1d4dc', size: 8 }, margin: { t: 30, b: 25, l: 35, r: 10 }, xaxis: { showgrid: false }, yaxis: { range: [0, 1.1], gridcolor: '#2a2e39' }, legend: { orientation: 'h', y: -0.2, font: { size: 7 } }, showlegend: true }}
                        config={{ displayModeBar: false }}
                        style={{ width: '100%', height: '180px' }}
                      />
                    </div>
                  ))}
                </div>
              ) : <div style={emptyState}>Brak danych MF</div>
            )}

            {/* IMPORTANCE */}
            {activeTab === 'importance' && (
              Object.keys(getImportance()).length > 0 ? (
                <Plot
                  data={[{
                    x: Object.values(getImportance()), y: Object.keys(getImportance()),
                    type: 'bar', orientation: 'h',
                    marker: { color: Object.values(getImportance()).map(v => `rgba(41,98,255,${0.3 + v / 100 * 0.7})`) },
                    text: Object.values(getImportance()).map(v => `${fmt(v, 1)}%`), textposition: 'outside'
                  }]}
                  layout={{ title: { text: 'Feature Importance', font: { color: '#d1d4dc', size: 14 } }, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#d1d4dc' }, margin: { t: 40, b: 40, l: 140, r: 50 }, xaxis: { title: '%', gridcolor: '#2a2e39' } }}
                  style={{ width: '100%', height: Math.max(280, Object.keys(getImportance()).length * 35) + 'px' }}
                />
              ) : <div style={emptyState}>Brak danych ważności</div>
            )}

            {/* RULES */}
            {activeTab === 'rules' && (
              getRules().length > 0 ? (
                <div style={{ maxHeight: '350px', overflowY: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8em' }}>
                    <thead style={{ position: 'sticky', top: 0, background: '#1a1d26' }}>
                      <tr style={{ color: '#888' }}><th style={{ padding: '8px', textAlign: 'left' }}>#</th><th style={{ padding: '8px', textAlign: 'left' }}>Reguła</th></tr>
                    </thead>
                    <tbody>
                      {getRules().map((rule, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid #2a2e39' }}>
                          <td style={{ padding: '6px', color: '#cba6f7' }}>{i + 1}</td>
                          <td style={{ padding: '6px', fontFamily: 'monospace', fontSize: '0.85em' }}>
                            <span style={{ color: '#fab387' }}>IF </span>
                            <span style={{ color: '#89b4fa' }}>{rule.antecedent || 'N/A'}</span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : <div style={emptyState}>Brak reguł</div>
            )}

            {/* DEBUG */}
            {activeTab === 'debug' && (
              <div style={{ maxHeight: '350px', overflowY: 'auto' }}>
                <h4 style={{ color: '#d1d4dc', marginTop: 0 }}>🐛 Debug Info</h4>
                <p style={{ color: '#888', fontSize: '0.85em' }}>Status: <strong>{results?.status || 'unknown'}</strong></p>
                <p style={{ color: '#888', fontSize: '0.85em' }}>Metrics keys: <strong>{Object.keys(results?.metrics || {}).join(', ') || 'none'}</strong></p>
                <p style={{ color: '#888', fontSize: '0.85em' }}>Predictions count: <strong>{getPredictions().dates?.length || 0}</strong></p>
                <p style={{ color: '#888', fontSize: '0.85em' }}>History epochs: <strong>{getHistory().epochs?.length || 0}</strong></p>
                <pre style={{ background: '#0d0d0d', padding: '10px', borderRadius: '4px', fontSize: '0.7em', color: '#888', overflow: 'auto', maxHeight: '200px' }}>
                  {rawResponse || JSON.stringify(results, null, 2).slice(0, 3000)}
                </pre>
              </div>
            )}
          </div>

          {/* MODEL SUMMARY */}
          <div style={{ marginTop: '15px', padding: '12px', background: '#1a1d26', borderRadius: '8px', border: '1px solid #363a45', fontSize: '0.8em' }}>
            <strong style={{ color: '#d1d4dc' }}>📊 Model:</strong>{' '}
            <span style={{ color: '#888' }}>
              {predictionType} | {mfType} | {optimizer} | 
              Rules: {results?.model_summary?.num_rules || '?'} | 
              Features: {(results?.model_summary?.feature_names || []).join(', ') || '?'}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

// ========== STYLE ==========

const labelStyle = { fontSize: '0.7em', color: '#888', marginBottom: '3px', display: 'block' };
const inputStyle = { width: '100%', padding: '6px', borderRadius: '4px', border: '1px solid #363a45', background: '#131722', color: '#fff', fontSize: '0.85em' };
const statBox = (highlight = false) => ({ background: '#2a2e39', padding: '15px', borderRadius: '8px', textAlign: 'center', border: `1px solid ${highlight ? '#00e676' : '#363a45'}` });
const statLabel = { color: '#888', fontSize: '0.8em', marginBottom: '5px' };
const statVal = { fontSize: '1.5em', fontWeight: 'bold', color: '#d1d4dc' };
const emptyState = { textAlign: 'center', padding: '60px 20px', color: '#666' };

export default AnfisMLPanel;
