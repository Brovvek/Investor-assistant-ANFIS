import React, { useState } from 'react';

const TrainingPanel = ({ config, ticker }) => {
  const [training, setTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [epochs, setEpochs] = useState(50);
  const [trainedParams, setTrainedParams] = useState(null);

  const startTraining = async () => {
    setTraining(true);
    setProgress(0);
    setResults(null);

    try {
      const response = await fetch('http://127.0.0.1:5000/api/train_anfis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker, config, epochs })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(l => l.trim());

        for (const line of lines) {
          try {
            const data = JSON.parse(line);

            if (data.status === 'progress') {
              setProgress(data.value);
            } else if (data.status === 'done') {
              setResults(data.result);
              setProgress(100);
              
              // Pobierz nauczone parametry
              const paramsRes = await fetch('http://127.0.0.1:5000/api/get_anfis_params');
              const paramsData = await paramsRes.json();
              setTrainedParams(paramsData);
            } else if (data.status === 'error') {
              alert('Błąd treningu: ' + data.message);
            }
          } catch (e) {
            console.error('Parse error:', e);
          }
        }
      }
    } catch (err) {
      alert('Błąd połączenia: ' + err.message);
    } finally {
      setTraining(false);
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>🧠 Trening ANFIS - Backpropagation</h3>
        <p style={styles.subtitle}>
          System uczy się optymalnych parametrów funkcji Gaussa (centra, sigmy) i wag konsekwentów
        </p>
      </div>

      {/* Kontrolki */}
      <div style={styles.controls}>
        <div style={styles.inputGroup}>
          <label style={styles.label}>Liczba epok:</label>
          <input
            type="number"
            value={epochs}
            onChange={(e) => setEpochs(parseInt(e.target.value))}
            min="10"
            max="200"
            style={styles.input}
            disabled={training}
          />
        </div>

        <button
          onClick={startTraining}
          disabled={training || Object.keys(config).filter(k => config[k].enabled).length === 0}
          style={{
            ...styles.button,
            opacity: training ? 0.5 : 1,
            cursor: training ? 'not-allowed' : 'pointer'
          }}
        >
          {training ? `🔄 Trening... ${progress}%` : '🚀 Rozpocznij Trening'}
        </button>
      </div>

      {/* Progress Bar */}
      {training && (
        <div style={styles.progressContainer}>
          <div style={styles.progressBar}>
            <div style={{ ...styles.progressFill, width: `${progress}%` }} />
          </div>
          <span style={styles.progressText}>{progress}%</span>
        </div>
      )}

      {/* Wyniki treningu */}
      {results && (
        <div style={styles.results}>
          <h4 style={styles.sectionTitle}>📊 Wyniki Treningu</h4>
          
          <div style={styles.metricsGrid}>
            <div style={styles.metricBox}>
              <div style={styles.metricLabel}>Train MSE</div>
              <div style={styles.metricValue}>{results.train_mse.toFixed(2)}</div>
            </div>
            
            <div style={styles.metricBox}>
              <div style={styles.metricLabel}>Test MSE</div>
              <div style={styles.metricValue}>{results.test_mse.toFixed(2)}</div>
            </div>
            
            <div style={styles.metricBox}>
              <div style={styles.metricLabel}>Overfitting Ratio</div>
              <div style={{
                ...styles.metricValue,
                color: results.overfitting_ratio < 1.2 ? '#66bb6a' : 
                       results.overfitting_ratio < 1.5 ? '#ffa726' : '#ef5350'
              }}>
                {results.overfitting_ratio.toFixed(2)}x
              </div>
              <div style={styles.metricHint}>
                {results.overfitting_ratio < 1.2 ? '✓ Brak overfittingu' : 
                 results.overfitting_ratio < 1.5 ? '⚠ Lekki overfitting' : '❌ Silny overfitting'}
              </div>
            </div>
          </div>

          {/* Wykres MSE */}
          {results.history && (
            <div style={styles.chartContainer}>
              <h5 style={styles.chartTitle}>Spadek błędu podczas treningu</h5>
              <MSEChart data={results.history} />
            </div>
          )}
        </div>
      )}

      {/* Nauczone parametry */}
      {trainedParams && (
        <div style={styles.params}>
          <h4 style={styles.sectionTitle}>🎯 Nauczone Parametry ANFIS</h4>
          
          {Object.keys(trainedParams).map(feature => (
            <div key={feature} style={styles.featureParams}>
              <h5 style={styles.featureName}>{feature}</h5>
              
              <div style={styles.paramsGrid}>
                <div style={styles.paramBlock}>
                  <strong>Centra (μ):</strong>
                  <div style={styles.paramValues}>
                    Low: {trainedParams[feature].centers[0].toFixed(1)} |
                    Med: {trainedParams[feature].centers[1].toFixed(1)} |
                    High: {trainedParams[feature].centers[2].toFixed(1)}
                  </div>
                </div>
                
                <div style={styles.paramBlock}>
                  <strong>Sigmy (σ):</strong>
                  <div style={styles.paramValues}>
                    σ₁: {trainedParams[feature].sigmas[0].toFixed(2)} |
                    σ₂: {trainedParams[feature].sigmas[1].toFixed(2)} |
                    σ₃: {trainedParams[feature].sigmas[2].toFixed(2)}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Info box */}
      <div style={styles.infoBox}>
        <strong>Jak działa trening:</strong>
        <br/>1. <strong>Forward Pass:</strong> Oblicza wyjście przez 5 warstw ANFIS
        <br/>2. <strong>Błąd:</strong> MSE = (output - target)²
        <br/>3. <strong>Backward Pass:</strong> Oblicza gradienty ∂E/∂center, ∂E/∂sigma, ∂E/∂w
        <br/>4. <strong>Update:</strong> param ← param - lr × gradient
        <br/>5. <strong>Walidacja:</strong> Test MSE sprawdza czy nie ma overfittingu
      </div>
    </div>
  );
};

// Komponent wykresu MSE
const MSEChart = ({ data }) => {
  const canvasRef = React.useRef(null);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear
    ctx.fillStyle = '#1a1d26';
    ctx.fillRect(0, 0, width, height);

    // Margins
    const margin = { top: 20, right: 20, bottom: 40, left: 60 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    const epochs = data.epochs;
    const mse = data.mse;

    const maxMSE = Math.max(...mse) * 1.1;
    const minMSE = 0;

    // Scale functions
    const scaleX = (x) => margin.left + (x / (epochs.length - 1)) * chartWidth;
    const scaleY = (y) => margin.top + chartHeight - ((y - minMSE) / (maxMSE - minMSE)) * chartHeight;

    // Grid
    ctx.strokeStyle = '#2a2e39';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
      const y = margin.top + (chartHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(width - margin.right, y);
      ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = '#888';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, height - margin.bottom);
    ctx.lineTo(width - margin.right, height - margin.bottom);
    ctx.stroke();

    // Labels
    ctx.fillStyle = '#888';
    ctx.font = '10px Arial';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const val = maxMSE - (maxMSE / 5) * i;
      const y = margin.top + (chartHeight / 5) * i;
      ctx.fillText(val.toFixed(0), margin.left - 5, y + 3);
    }

    ctx.textAlign = 'center';
    ctx.fillText('Epoka', width / 2, height - 5);

    // Plot line with gradient
    const gradient = ctx.createLinearGradient(0, scaleY(maxMSE), 0, scaleY(minMSE));
    gradient.addColorStop(0, '#ef5350');
    gradient.addColorStop(1, '#66bb6a');

    ctx.strokeStyle = gradient;
    ctx.lineWidth = 3;
    ctx.beginPath();
    
    epochs.forEach((epoch, i) => {
      const px = scaleX(i);
      const py = scaleY(mse[i]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();

    // Fill area
    ctx.fillStyle = 'rgba(239, 83, 80, 0.1)';
    ctx.beginPath();
    ctx.moveTo(scaleX(0), scaleY(0));
    epochs.forEach((epoch, i) => {
      ctx.lineTo(scaleX(i), scaleY(mse[i]));
    });
    ctx.lineTo(scaleX(epochs.length - 1), scaleY(0));
    ctx.closePath();
    ctx.fill();

  }, [data]);

  return <canvas ref={canvasRef} width={700} height={250} style={{ width: '100%', height: 'auto' }} />;
};

const styles = {
  container: {
    background: '#1e222d',
    borderRadius: '8px',
    padding: '20px',
    border: '1px solid #363a45'
  },
  header: {
    borderBottom: '2px solid #2962ff',
    paddingBottom: '15px',
    marginBottom: '20px'
  },
  title: {
    margin: 0,
    color: '#d1d4dc',
    fontSize: '1.3em'
  },
  subtitle: {
    margin: '5px 0 0 0',
    color: '#888',
    fontSize: '0.9em'
  },
  controls: {
    display: 'flex',
    gap: '15px',
    alignItems: 'center',
    marginBottom: '20px',
    padding: '15px',
    background: '#131722',
    borderRadius: '6px'
  },
  inputGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px'
  },
  label: {
    color: '#888',
    fontSize: '0.9em'
  },
  input: {
    padding: '8px 12px',
    borderRadius: '4px',
    border: '1px solid #363a45',
    background: '#1e222d',
    color: '#fff',
    width: '80px',
    fontSize: '0.9em'
  },
  button: {
    padding: '10px 25px',
    borderRadius: '6px',
    border: 'none',
    background: 'linear-gradient(135deg, #2962ff 0%, #89b4fa 100%)',
    color: '#fff',
    fontWeight: 'bold',
    fontSize: '0.95em',
    boxShadow: '0 4px 15px rgba(41, 98, 255, 0.3)',
    transition: 'all 0.2s'
  },
  progressContainer: {
    marginBottom: '20px'
  },
  progressBar: {
    width: '100%',
    height: '30px',
    background: '#1a1d26',
    borderRadius: '15px',
    overflow: 'hidden',
    position: 'relative'
  },
  progressFill: {
    height: '100%',
    background: 'linear-gradient(90deg, #2962ff 0%, #66bb6a 100%)',
    transition: 'width 0.3s ease'
  },
  progressText: {
    display: 'block',
    textAlign: 'center',
    marginTop: '5px',
    color: '#d1d4dc',
    fontWeight: 'bold'
  },
  results: {
    marginTop: '20px',
    padding: '15px',
    background: '#131722',
    borderRadius: '6px'
  },
  sectionTitle: {
    margin: '0 0 15px 0',
    color: '#d1d4dc',
    fontSize: '1.1em'
  },
  metricsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '15px',
    marginBottom: '20px'
  },
  metricBox: {
    background: '#2a2e39',
    padding: '15px',
    borderRadius: '6px',
    textAlign: 'center'
  },
  metricLabel: {
    color: '#888',
    fontSize: '0.85em',
    marginBottom: '5px'
  },
  metricValue: {
    color: '#d1d4dc',
    fontSize: '1.8em',
    fontWeight: 'bold'
  },
  metricHint: {
    color: '#888',
    fontSize: '0.75em',
    marginTop: '5px'
  },
  chartContainer: {
    marginTop: '20px',
    padding: '15px',
    background: '#1a1d26',
    borderRadius: '6px'
  },
  chartTitle: {
    margin: '0 0 15px 0',
    color: '#888',
    fontSize: '0.9em',
    textAlign: 'center'
  },
  params: {
    marginTop: '20px',
    padding: '15px',
    background: '#131722',
    borderRadius: '6px'
  },
  featureParams: {
    marginBottom: '15px',
    padding: '12px',
    background: '#2a2e39',
    borderRadius: '6px',
    borderLeft: '4px solid #2962ff'
  },
  featureName: {
    margin: '0 0 10px 0',
    color: '#89b4fa',
    fontSize: '0.95em'
  },
  paramsGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '15px'
  },
  paramBlock: {
    fontSize: '0.85em',
    color: '#b0b3b8'
  },
  paramValues: {
    marginTop: '5px',
    fontFamily: 'monospace',
    color: '#d1d4dc'
  },
  infoBox: {
    marginTop: '20px',
    background: '#2a2e39',
    padding: '15px',
    borderRadius: '6px',
    fontSize: '0.85em',
    lineHeight: '1.8',
    color: '#b0b3b8',
    borderLeft: '4px solid #66bb6a'
  }
};

export default TrainingPanel;