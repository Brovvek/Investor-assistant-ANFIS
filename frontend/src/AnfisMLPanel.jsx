import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';
import './App.css';

const AnfisMLPanel = ({ ticker, config }) => {
  // Training parameters
  const [epochs, setEpochs] = useState(100);
  const [numMfs, setNumMfs] = useState(3);
  const [batchSize, setBatchSize] = useState(64);
  const [learningRate, setLearningRate] = useState(0.01);
  const [mfType, setMfType] = useState('gauss');
  const [hybrid, setHybrid] = useState(true);
  const [optimizer, setOptimizer] = useState('adam');
  const [lookahead, setLookahead] = useState(1);
  
  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [liveMetrics, setLiveMetrics] = useState(null);
  
  // Results
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  
  // Active tab for results
  const [activeTab, setActiveTab] = useState('predictions');

  const startTraining = async () => {
    setIsTraining(true);
    setProgress(0);
    setCurrentEpoch(0);
    setResults(null);
    setError('');
    setLiveMetrics(null);

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
          lookahead: parseInt(lookahead)
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
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.trim()) {
            try {
              const data = JSON.parse(line);
              
              if (data.status === 'training') {
                setProgress(data.progress);
                setCurrentEpoch(data.epoch);
                setLiveMetrics({
                  train_loss: data.train_loss,
                  val_loss: data.val_loss,
                  val_rmse: data.val_rmse,
                  val_mape: data.val_mape
                });
              } else if (data.status === 'done') {
                setResults(data);
                setProgress(100);
              } else if (data.status === 'error') {
                setError(data.message);
              } else if (data.status === 'preparing' || data.status === 'data_ready' || data.status === 'model_built') {
                // Info messages
                console.log(data);
              }
            } catch (e) {
              console.error('Parse error:', e);
            }
          }
        }
      }
    } catch (err) {
      setError(err.message || 'Błąd treningu');
    } finally {
      setIsTraining(false);
    }
  };

  // Check if we have active features
  const activeFeatures = Object.entries(config).filter(([k, v]) => v.enabled);
  const canTrain = activeFeatures.length >= 1;

  return (
    <div style={{ padding: '10px' }}>
      {/* PARAMETRY TRENINGU */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', 
        gap: '15px',
        marginBottom: '20px',
        padding: '15px',
        background: '#1a1d26',
        borderRadius: '8px',
        border: '1px solid #363a45'
      }}>
        <div className="param-group">
          <label style={styles.label}>Epoki uczenia</label>
          <input 
            type="number" 
            value={epochs} 
            onChange={e => setEpochs(e.target.value)}
            min="10" max="1000" step="10"
            style={styles.input}
          />
        </div>
        
        <div className="param-group">
          <label style={styles.label}>Funkcje przynależności</label>
          <input 
            type="number" 
            value={numMfs} 
            onChange={e => setNumMfs(e.target.value)}
            min="2" max="7" step="1"
            style={styles.input}
          />
        </div>
        
        <div className="param-group">
          <label style={styles.label}>Typ MF</label>
          <select value={mfType} onChange={e => setMfType(e.target.value)} style={styles.input}>
            <option value="gauss">Gaussowska</option>
            <option value="bell">Dzwonowa (Bell)</option>
            <option value="tri">Trójkątna</option>
          </select>
        </div>
        
        <div className="param-group">
          <label style={styles.label}>Optymalizator</label>
          <select value={optimizer} onChange={e => setOptimizer(e.target.value)} style={styles.input}>
            <option value="adam">Adam</option>
            <option value="sgd">SGD + Momentum</option>
            <option value="rprop">Rprop</option>
          </select>
        </div>
        
        <div className="param-group">
          <label style={styles.label}>Learning Rate</label>
          <input 
            type="number" 
            value={learningRate} 
            onChange={e => setLearningRate(e.target.value)}
            min="0.0001" max="0.1" step="0.001"
            style={styles.input}
          />
        </div>
        
        <div className="param-group">
          <label style={styles.label}>Batch Size</label>
          <select value={batchSize} onChange={e => setBatchSize(e.target.value)} style={styles.input}>
            <option value="32">32</option>
            <option value="64">64</option>
            <option value="128">128</option>
            <option value="256">256</option>
          </select>
        </div>
        
        <div className="param-group">
          <label style={styles.label}>Predykcja (dni)</label>
          <input 
            type="number" 
            value={lookahead} 
            onChange={e => setLookahead(e.target.value)}
            min="1" max="30" step="1"
            style={styles.input}
          />
        </div>
        
        <div className="param-group">
          <label style={styles.label}>
            <input 
              type="checkbox" 
              checked={hybrid} 
              onChange={e => setHybrid(e.target.checked)}
              style={{ marginRight: '8px' }}
            />
            Hybrid Learning (LSE)
          </label>
        </div>
      </div>

      {/* PRZYCISK STARTU */}
      <div style={{ display: 'flex', gap: '15px', alignItems: 'center', marginBottom: '20px' }}>
        <button 
          onClick={startTraining} 
          disabled={isTraining || !canTrain}
          className="ai-button"
          style={{ 
            padding: '12px 30px', 
            fontSize: '1em',
            background: canTrain ? 'linear-gradient(135deg, #2962ff 0%, #00c853 100%)' : '#444'
          }}
        >
          {isTraining ? `Trenuję... ${progress}%` : '🧠 Rozpocznij Uczenie ANFIS'}
        </button>
        
        {!canTrain && (
          <span style={{ color: '#ff5252', fontSize: '0.9em' }}>
            ⚠️ Wybierz przynajmniej 1 wskaźnik w panelu "Aktywne Wskaźniki"
          </span>
        )}
        
        {isTraining && liveMetrics && (
          <div style={{ display: 'flex', gap: '20px', fontSize: '0.85em', color: '#888' }}>
            <span>Epoka: {currentEpoch}/{epochs}</span>
            <span>Val RMSE: {liveMetrics.val_rmse?.toFixed(4)}</span>
            <span>MAPE: {liveMetrics.val_mape?.toFixed(2)}%</span>
          </div>
        )}
      </div>

      {/* PASEK POSTĘPU */}
      {isTraining && (
        <div style={{ 
          height: '8px', 
          background: '#2a2e39', 
          borderRadius: '4px', 
          marginBottom: '20px',
          overflow: 'hidden'
        }}>
          <div style={{ 
            height: '100%', 
            width: `${progress}%`, 
            background: 'linear-gradient(90deg, #2962ff, #00c853)',
            transition: 'width 0.3s ease'
          }} />
        </div>
      )}

      {/* BŁĄD */}
      {error && (
        <div style={{ 
          background: 'rgba(255,82,82,0.1)', 
          border: '1px solid #ff5252',
          borderRadius: '8px',
          padding: '15px',
          marginBottom: '20px',
          color: '#ff5252'
        }}>
          ❌ {error}
        </div>
      )}

      {/* WYNIKI */}
      {results && results.status === 'done' && (
        <div>
          {/* METRYKI */}
          <div className="stats-grid" style={{ marginBottom: '20px' }}>
            <div className="stat-box" style={{ borderColor: '#2962ff' }}>
              <div className="stat-label">Test RMSE</div>
              <div className="stat-val" style={{ color: '#89b4fa' }}>
                {results.metrics.test_rmse.toFixed(4)}
              </div>
              <div className="stat-sub">Root Mean Square Error</div>
            </div>
            
            <div className="stat-box" style={{ borderColor: results.metrics.test_mape < 5 ? '#00e676' : '#fab387' }}>
              <div className="stat-label">MAPE</div>
              <div className={`stat-val ${results.metrics.test_mape < 5 ? 'green' : ''}`}>
                {results.metrics.test_mape.toFixed(2)}%
              </div>
              <div className="stat-sub">Mean Absolute Percentage Error</div>
            </div>
            
            <div className="stat-box" style={{ borderColor: results.metrics.direction_accuracy > 55 ? '#00e676' : '#888' }}>
              <div className="stat-label">Celność Kierunku</div>
              <div className={`stat-val ${results.metrics.direction_accuracy > 55 ? 'green' : ''}`}>
                {results.metrics.direction_accuracy.toFixed(1)}%
              </div>
              <div className="stat-sub">Direction Accuracy</div>
            </div>
            
            <div className="stat-box">
              <div className="stat-label">Reguły ANFIS</div>
              <div className="stat-val" style={{ color: '#cba6f7' }}>
                {results.model_summary.num_rules}
              </div>
              <div className="stat-sub">{results.model_summary.num_inputs} wejść × {numMfs} MF</div>
            </div>
          </div>

          {/* TABS */}
          <div style={{ 
            display: 'flex', 
            gap: '5px', 
            marginBottom: '15px',
            borderBottom: '1px solid #363a45',
            paddingBottom: '10px'
          }}>
            {['predictions', 'loss', 'mf', 'importance', 'rules'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                style={{
                  padding: '8px 16px',
                  background: activeTab === tab ? '#2962ff' : 'transparent',
                  border: activeTab === tab ? 'none' : '1px solid #363a45',
                  borderRadius: '4px',
                  color: activeTab === tab ? '#fff' : '#888',
                  cursor: 'pointer',
                  fontSize: '0.9em'
                }}
              >
                {tab === 'predictions' && '📈 Predykcje'}
                {tab === 'loss' && '📉 Historia Błędu'}
                {tab === 'mf' && '🔔 Funkcje Przynależności'}
                {tab === 'importance' && '⚖️ Ważność Cech'}
                {tab === 'rules' && '📜 Reguły'}
              </button>
            ))}
          </div>

          {/* TAB CONTENT */}
          <div style={{ 
            background: '#1e222d', 
            borderRadius: '8px', 
            padding: '15px',
            border: '1px solid #363a45'
          }}>
            
            {/* PREDYKCJE */}
            {activeTab === 'predictions' && (
              <Plot
                data={[
                  {
                    x: results.predictions.dates,
                    y: results.predictions.actual,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Rzeczywista Cena',
                    line: { color: '#2962ff', width: 2 }
                  },
                  {
                    x: results.predictions.dates,
                    y: results.predictions.predicted,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Predykcja ANFIS',
                    line: { color: '#00e676', width: 2, dash: 'dot' }
                  }
                ]}
                layout={{
                  title: { text: `Predykcja Ceny (${lookahead} dzień do przodu)`, font: { color: '#d1d4dc', size: 14 } },
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                  font: { color: '#d1d4dc' },
                  margin: { t: 40, b: 40, l: 60, r: 20 },
                  xaxis: { showgrid: false },
                  yaxis: { title: 'Cena', gridcolor: '#2a2e39' },
                  legend: { orientation: 'h', y: 1.1 },
                  hovermode: 'x unified'
                }}
                useResizeHandler={true}
                style={{ width: '100%', height: '400px' }}
              />
            )}

            {/* HISTORIA BŁĘDU */}
            {activeTab === 'loss' && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
                <Plot
                  data={[
                    {
                      x: results.training_history.epochs,
                      y: results.training_history.train_loss,
                      type: 'scatter',
                      mode: 'lines',
                      name: 'Train Loss',
                      line: { color: '#2962ff', width: 2 }
                    },
                    {
                      x: results.training_history.epochs,
                      y: results.training_history.val_loss,
                      type: 'scatter',
                      mode: 'lines',
                      name: 'Validation Loss',
                      line: { color: '#ff5252', width: 2 }
                    }
                  ]}
                  layout={{
                    title: { text: 'Loss (MSE)', font: { color: '#d1d4dc', size: 12 } },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#d1d4dc', size: 10 },
                    margin: { t: 40, b: 40, l: 50, r: 20 },
                    xaxis: { title: 'Epoka', showgrid: false },
                    yaxis: { gridcolor: '#2a2e39' },
                    legend: { orientation: 'h', y: 1.15 }
                  }}
                  style={{ width: '100%', height: '300px' }}
                />
                
                <Plot
                  data={[
                    {
                      x: results.training_history.epochs,
                      y: results.training_history.val_mape,
                      type: 'scatter',
                      mode: 'lines',
                      name: 'MAPE (%)',
                      line: { color: '#00e676', width: 2 },
                      fill: 'tozeroy',
                      fillcolor: 'rgba(0,230,118,0.1)'
                    }
                  ]}
                  layout={{
                    title: { text: 'MAPE (%)', font: { color: '#d1d4dc', size: 12 } },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#d1d4dc', size: 10 },
                    margin: { t: 40, b: 40, l: 50, r: 20 },
                    xaxis: { title: 'Epoka', showgrid: false },
                    yaxis: { gridcolor: '#2a2e39' },
                    legend: { orientation: 'h', y: 1.15 }
                  }}
                  style={{ width: '100%', height: '300px' }}
                />
              </div>
            )}

            {/* FUNKCJE PRZYNALEŻNOŚCI */}
            {activeTab === 'mf' && (
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', 
                gap: '15px' 
              }}>
                {Object.entries(results.membership_functions).map(([varName, data]) => (
                  <div key={varName} style={{ background: '#1a1d26', borderRadius: '8px', padding: '10px' }}>
                    <Plot
                      data={Object.entries(data.mfs).map(([mfName, yValues], idx) => ({
                        x: data.x,
                        y: yValues,
                        type: 'scatter',
                        mode: 'lines',
                        name: mfName,
                        line: { 
                          color: ['#2962ff', '#00e676', '#ff5252', '#fab387', '#cba6f7', '#f9e2af', '#89b4fa'][idx % 7],
                          width: 2 
                        }
                      }))}
                      layout={{
                        title: { 
                          text: varName, 
                          font: { color: '#d1d4dc', size: 12 } 
                        },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { color: '#d1d4dc', size: 9 },
                        margin: { t: 35, b: 30, l: 40, r: 10 },
                        xaxis: { showgrid: false },
                        yaxis: { 
                          title: 'μ', 
                          range: [0, 1.1],
                          gridcolor: '#2a2e39' 
                        },
                        legend: { orientation: 'h', y: -0.15, font: { size: 8 } },
                        showlegend: true
                      }}
                      config={{ displayModeBar: false }}
                      style={{ width: '100%', height: '220px' }}
                    />
                    <div style={{ fontSize: '0.75em', color: '#666', marginTop: '5px' }}>
                      {data.params.map((p, i) => (
                        <span key={i} style={{ marginRight: '10px' }}>
                          {p.name}: {Object.entries(p.params).map(([k, v]) => `${k}=${v.toFixed(3)}`).join(', ')}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* WAŻNOŚĆ CECH */}
            {activeTab === 'importance' && results.feature_importance && (
              <div>
                <Plot
                  data={[{
                    x: Object.values(results.feature_importance),
                    y: Object.keys(results.feature_importance),
                    type: 'bar',
                    orientation: 'h',
                    marker: {
                      color: Object.values(results.feature_importance).map(v => 
                        `rgba(41, 98, 255, ${0.3 + v / 100 * 0.7})`
                      )
                    },
                    text: Object.values(results.feature_importance).map(v => `${v.toFixed(1)}%`),
                    textposition: 'outside'
                  }]}
                  layout={{
                    title: { text: 'Ważność Cech (Permutation Importance)', font: { color: '#d1d4dc', size: 14 } },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#d1d4dc' },
                    margin: { t: 40, b: 40, l: 150, r: 60 },
                    xaxis: { title: 'Importance (%)', showgrid: true, gridcolor: '#2a2e39' },
                    yaxis: { automargin: true }
                  }}
                  style={{ width: '100%', height: Math.max(300, Object.keys(results.feature_importance).length * 40) + 'px' }}
                />
              </div>
            )}

            {/* REGUŁY */}
            {activeTab === 'rules' && (
              <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85em' }}>
                  <thead style={{ position: 'sticky', top: 0, background: '#1a1d26' }}>
                    <tr style={{ color: '#888', borderBottom: '1px solid #363a45' }}>
                      <th style={{ padding: '10px', textAlign: 'left' }}>#</th>
                      <th style={{ padding: '10px', textAlign: 'left' }}>Reguła (IF ... THEN)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.rules.map((rule, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #2a2e39' }}>
                        <td style={{ padding: '8px', color: '#cba6f7' }}>{rule.id + 1}</td>
                        <td style={{ padding: '8px', fontFamily: 'monospace', fontSize: '0.9em' }}>
                          <span style={{ color: '#fab387' }}>IF </span>
                          {rule.antecedent.split(' and ').map((part, j) => (
                            <span key={j}>
                              {j > 0 && <span style={{ color: '#fab387' }}> AND </span>}
                              <span style={{ color: '#89b4fa' }}>{part}</span>
                            </span>
                          ))}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {results.rules.length >= 20 && (
                  <div style={{ textAlign: 'center', padding: '10px', color: '#666', fontSize: '0.8em' }}>
                    Pokazano pierwsze 20 z {results.model_summary.num_rules} reguł
                  </div>
                )}
              </div>
            )}
          </div>

          {/* PODSUMOWANIE MODELU */}
          <div style={{ 
            marginTop: '20px', 
            padding: '15px', 
            background: '#1a1d26', 
            borderRadius: '8px',
            border: '1px solid #363a45',
            fontSize: '0.85em'
          }}>
            <h4 style={{ margin: '0 0 10px 0', color: '#d1d4dc' }}>📊 Podsumowanie Modelu</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px' }}>
              <div><span style={{ color: '#888' }}>Architektura:</span> ANFIS ({mfType === 'gauss' ? 'Gaussian' : mfType === 'bell' ? 'Bell' : 'Triangular'} MF)</div>
              <div><span style={{ color: '#888' }}>Uczenie:</span> {hybrid ? 'Hybrydowe (LSE + Backprop)' : 'Backpropagation'}</div>
              <div><span style={{ color: '#888' }}>Optymalizator:</span> {optimizer.toUpperCase()}</div>
              <div><span style={{ color: '#888' }}>Próbki treningowe:</span> {results.metrics.train_samples}</div>
              <div><span style={{ color: '#888' }}>Próbki testowe:</span> {results.metrics.test_samples}</div>
              <div><span style={{ color: '#888' }}>Cechy:</span> {results.model_summary.feature_names?.join(', ')}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const styles = {
  label: {
    fontSize: '0.8em',
    color: '#888',
    marginBottom: '5px',
    display: 'block'
  },
  input: {
    width: '100%',
    padding: '8px',
    borderRadius: '4px',
    border: '1px solid #363a45',
    background: '#131722',
    color: '#fff',
    fontSize: '0.9em'
  }
};

export default AnfisMLPanel;
