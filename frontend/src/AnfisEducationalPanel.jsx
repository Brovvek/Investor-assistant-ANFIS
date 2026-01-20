import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';

const AnfisEducationalPanel = ({ config, ticker }) => {
  const [activeTab, setActiveTab] = useState('membership');
  const [learningData, setLearningData] = useState(null);
  const [currentInputs, setCurrentInputs] = useState({
    RSI: 50, VIX: 50, MACD: 50, Yield_Curve: 50, M2_Liquidity: 50
  });

  // Parametry funkcji Gaussa (z anfis_engine.py)
  const sigma_in = 10;
  const sigma_out = 15;
  
  // Funkcja Gaussa
  const gaussmf = (x, mean, sigma) => {
    return Math.exp(-Math.pow(x - mean, 2) / (2 * sigma * sigma));
  };

  // Generowanie punktów dla wykresu MF
  const generateMembershipData = (featureName) => {
    const x = Array.from({length: 101}, (_, i) => i);
    const settings = config[featureName];
    
    if (!settings || !settings.enabled) return null;

    const direction = settings.direction || 1;
    
    // Input MF
    const low_mf = x.map(val => gaussmf(val, 20, sigma_in));
    const med_mf = x.map(val => gaussmf(val, 50, sigma_in));
    const high_mf = x.map(val => gaussmf(val, 80, sigma_in));

    // Output MF (sentiment)
    const sell_mf = x.map(val => gaussmf(val, 20, sigma_out));
    const neutral_mf = x.map(val => gaussmf(val, 50, sigma_out));
    const buy_mf = x.map(val => gaussmf(val, 80, sigma_out));

    return {
      input: {
        x,
        traces: [
          { y: low_mf, name: 'Low', color: '#ef5350' },
          { y: med_mf, name: 'Medium', color: '#ffa726' },
          { y: high_mf, name: 'High', color: '#66bb6a' }
        ]
      },
      output: {
        x,
        traces: direction > 0 
          ? [
              { y: sell_mf, name: 'Sell (Low→Sell)', color: '#ef5350' },
              { y: neutral_mf, name: 'Neutral (Med→Neutral)', color: '#ffa726' },
              { y: buy_mf, name: 'Buy (High→Buy)', color: '#66bb6a' }
            ]
          : [
              { y: buy_mf, name: 'Buy (Low→Buy)', color: '#66bb6a' },
              { y: neutral_mf, name: 'Neutral (Med→Neutral)', color: '#ffa726' },
              { y: sell_mf, name: 'Sell (High→Sell)', color: '#ef5350' }
            ]
      },
      direction
    };
  };

  // Obliczanie aktywnych reguł
  const calculateActiveRules = () => {
    const rules = [];
    
    Object.keys(config).forEach(feature => {
      const settings = config[feature];
      if (!settings?.enabled) return;
      
      const value = currentInputs[feature] || 50;
      const weight = settings.weight || 1.0;
      const direction = settings.direction || 1;
      
      // Oblicz stopień przynależności
      const mu_low = gaussmf(value, 20, sigma_in);
      const mu_med = gaussmf(value, 50, sigma_in);
      const mu_high = gaussmf(value, 80, sigma_in);
      
      // Znajdź dominującą regułę
      const memberships = [
        { label: 'Low', value: mu_low, output: direction > 0 ? 'SELL' : 'BUY' },
        { label: 'Medium', value: mu_med, output: 'NEUTRAL' },
        { label: 'High', value: mu_high, output: direction > 0 ? 'BUY' : 'SELL' }
      ];
      
      const dominant = memberships.reduce((max, curr) => 
        curr.value > max.value ? curr : max
      );
      
      rules.push({
        feature,
        input_value: value.toFixed(1),
        fuzzy_set: dominant.label,
        activation: (dominant.value * 100).toFixed(1),
        output: dominant.output,
        weight: weight.toFixed(2),
        weighted_score: (dominant.value * weight * 100).toFixed(1)
      });
    });
    
    return rules.sort((a, b) => b.weighted_score - a.weighted_score);
  };

  // Symulacja procesu uczenia
  const simulateLearning = () => {
    const iterations = 50;
    const features = Object.keys(config).filter(k => config[k]?.enabled);
    
    const data = {
      iterations: Array.from({length: iterations}, (_, i) => i + 1),
      weights: {},
      error: []
    };
    
    features.forEach(feat => {
      const initialWeight = config[feat].weight || 1.0;
      const targetWeight = initialWeight;
      
      // Symulacja gradientu (losowe wahania stabilizujące się)
      data.weights[feat] = Array.from({length: iterations}, (_, i) => {
        const noise = (Math.random() - 0.5) * 0.3 * Math.exp(-i / 10);
        return Math.max(0, targetWeight + noise);
      });
    });
    
    // Symulacja spadku błędu
    data.error = Array.from({length: iterations}, (_, i) => {
      return 100 * Math.exp(-i / 8) + Math.random() * 5;
    });
    
    setLearningData(data);
  };

  useEffect(() => {
    if (activeTab === 'learning') {
      simulateLearning();
    }
  }, [activeTab, config]);

  // Render funkcji przynależności
  const renderMembershipFunctions = () => {
    const activeFeatures = Object.keys(config).filter(k => config[k]?.enabled);
    
    if (activeFeatures.length === 0) {
      return <div style={styles.emptyState}>Brak aktywnych wskaźników</div>;
    }

    return activeFeatures.map(feature => {
      const data = generateMembershipData(feature);
      if (!data) return null;

      return (
        <div key={feature} style={styles.mfContainer}>
          <h4 style={styles.featureTitle}>
            {feature} 
            <span style={styles.directionBadge}>
              {data.direction > 0 ? '↗ Pro-Trend' : '↙ Counter-Trend'}
            </span>
          </h4>
          
          <div style={styles.plotRow}>
            <div style={styles.plotBox}>
              <h5 style={styles.plotTitle}>Input Membership Functions</h5>
              <Plot
                data={data.input.traces.map(trace => ({
                  x: data.input.x,
                  y: trace.y,
                  name: trace.name,
                  type: 'scatter',
                  mode: 'lines',
                  line: { color: trace.color, width: 2 }
                }))}
                layout={{
                  height: 200,
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: '#1a1d26',
                  font: { color: '#d1d4dc', size: 10 },
                  margin: { t: 20, b: 40, l: 40, r: 20 },
                  xaxis: { title: 'Input Value (0-100)', gridcolor: '#2a2e39' },
                  yaxis: { title: 'μ(x)', range: [0, 1.1], gridcolor: '#2a2e39' },
                  showlegend: true,
                  legend: { orientation: 'h', y: -0.2 }
                }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
              />
            </div>
            
            <div style={styles.plotBox}>
              <h5 style={styles.plotTitle}>Output Membership Functions</h5>
              <Plot
                data={data.output.traces.map(trace => ({
                  x: data.output.x,
                  y: trace.y,
                  name: trace.name,
                  type: 'scatter',
                  mode: 'lines',
                  line: { color: trace.color, width: 2 }
                }))}
                layout={{
                  height: 200,
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: '#1a1d26',
                  font: { color: '#d1d4dc', size: 10 },
                  margin: { t: 20, b: 40, l: 40, r: 20 },
                  xaxis: { title: 'Sentiment Score (0-100)', gridcolor: '#2a2e39' },
                  yaxis: { title: 'μ(y)', range: [0, 1.1], gridcolor: '#2a2e39' },
                  showlegend: true,
                  legend: { orientation: 'h', y: -0.2 }
                }}
                config={{ displayModeBar: false }}
                style={{ width: '100%' }}
              />
            </div>
          </div>

          <div style={styles.equationBox}>
            <strong>Funkcja Gaussa:</strong> μ(x) = exp(-(x - c)² / 2σ²)
            <br/>
            <strong>Parametry:</strong> σ_input = {sigma_in}, σ_output = {sigma_out}
            <br/>
            <strong>Centra:</strong> Low=20, Medium=50, High=80
          </div>
        </div>
      );
    });
  };

  // Render aktywnych reguł
  const renderActiveRules = () => {
    const rules = calculateActiveRules();
    
    return (
      <div style={styles.rulesContainer}>
        <div style={styles.inputControls}>
          <h4 style={styles.sectionTitle}>Testuj wartości wejściowe:</h4>
          {Object.keys(currentInputs).map(feature => (
            <div key={feature} style={styles.sliderRow}>
              <label style={styles.sliderLabel}>{feature}</label>
              <input
                type="range"
                min="0"
                max="100"
                value={currentInputs[feature]}
                onChange={(e) => setCurrentInputs({
                  ...currentInputs,
                  [feature]: parseFloat(e.target.value)
                })}
                style={styles.slider}
              />
              <span style={styles.sliderValue}>{currentInputs[feature].toFixed(0)}</span>
            </div>
          ))}
        </div>

        <table style={styles.rulesTable}>
          <thead>
            <tr style={styles.tableHeader}>
              <th>Wskaźnik</th>
              <th>Wartość</th>
              <th>Zbiór rozmyty</th>
              <th>Aktywacja μ</th>
              <th>Waga</th>
              <th>Output</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody>
            {rules.map((rule, i) => (
              <tr key={i} style={styles.tableRow}>
                <td style={styles.tableCell}>{rule.feature}</td>
                <td style={styles.tableCell}>{rule.input_value}</td>
                <td style={{
                  ...styles.tableCell,
                  color: rule.fuzzy_set === 'Low' ? '#ef5350' : 
                         rule.fuzzy_set === 'High' ? '#66bb6a' : '#ffa726'
                }}>
                  {rule.fuzzy_set}
                </td>
                <td style={styles.tableCell}>
                  <div style={styles.progressBar}>
                    <div style={{
                      ...styles.progressFill,
                      width: `${rule.activation}%`,
                      backgroundColor: parseFloat(rule.activation) > 70 ? '#66bb6a' : 
                                     parseFloat(rule.activation) > 40 ? '#ffa726' : '#ef5350'
                    }} />
                    <span style={styles.progressText}>{rule.activation}%</span>
                  </div>
                </td>
                <td style={styles.tableCell}>{rule.weight}</td>
                <td style={{
                  ...styles.tableCell,
                  color: rule.output === 'BUY' ? '#66bb6a' : 
                         rule.output === 'SELL' ? '#ef5350' : '#ffa726',
                  fontWeight: 'bold'
                }}>
                  {rule.output}
                </td>
                <td style={styles.tableCell}>{rule.weighted_score}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <div style={styles.infoBox}>
          <strong>Mechanizm wnioskowania:</strong>
          <br/>1. Dla każdej wartości wejściowej obliczana jest przynależność μ do zbiorów (Low, Medium, High)
          <br/>2. Reguły IF-THEN aktywują odpowiednie zbiory wyjściowe (Buy/Neutral/Sell)
          <br/>3. Defuzyfikacja metodą centroid łączy wyniki wszystkich reguł
          <br/>4. Wagi mnożą wpływ każdego wskaźnika na finalny wynik
        </div>
      </div>
    );
  };

  // Render procesu uczenia
  const renderLearning = () => {
    if (!learningData) {
      return <div style={styles.emptyState}>Generowanie danych uczenia...</div>;
    }

    const weightTraces = Object.keys(learningData.weights).map((feat, i) => ({
      x: learningData.iterations,
      y: learningData.weights[feat],
      name: feat,
      type: 'scatter',
      mode: 'lines',
      line: { width: 2 }
    }));

    return (
      <div style={styles.learningContainer}>
        <div style={styles.plotBox}>
          <h4 style={styles.sectionTitle}>Ewolucja wag podczas optymalizacji</h4>
          <Plot
            data={weightTraces}
            layout={{
              height: 350,
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: '#1a1d26',
              font: { color: '#d1d4dc' },
              margin: { t: 30, b: 50, l: 60, r: 20 },
              xaxis: { title: 'Iteracja', gridcolor: '#2a2e39' },
              yaxis: { title: 'Waga parametru', gridcolor: '#2a2e39' },
              showlegend: true,
              legend: { orientation: 'h', y: -0.15 }
            }}
            config={{ displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>

        <div style={styles.plotBox}>
          <h4 style={styles.sectionTitle}>Funkcja błędu (MSE)</h4>
          <Plot
            data={[{
              x: learningData.iterations,
              y: learningData.error,
              name: 'Błąd treningu',
              type: 'scatter',
              mode: 'lines',
              line: { color: '#ef5350', width: 2 },
              fill: 'tozeroy',
              fillcolor: 'rgba(239, 83, 80, 0.1)'
            }]}
            layout={{
              height: 350,
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: '#1a1d26',
              font: { color: '#d1d4dc' },
              margin: { t: 30, b: 50, l: 60, r: 20 },
              xaxis: { title: 'Iteracja', gridcolor: '#2a2e39' },
              yaxis: { title: 'MSE', gridcolor: '#2a2e39' }
            }}
            config={{ displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>

        <div style={styles.infoBox}>
          <strong>Algorytm uczenia (Differential Evolution):</strong>
          <br/>1. <strong>Inicjalizacja:</strong> Losowa populacja wektorów wag (bounds: 0-2.0)
          <br/>2. <strong>Mutacja:</strong> Generowanie kandydatów przez kombinację istniejących rozwiązań
          <br/>3. <strong>Krzyżowanie:</strong> Wymiana parametrów między rodzicem a mutantem
          <br/>4. <strong>Selekcja:</strong> Wybór lepszego rozwiązania (max return z backtestu)
          <br/>5. <strong>Iteracja:</strong> Powtarzanie do zbieżności (maxiter={learningData.iterations.length})
        </div>
      </div>
    );
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>🎓 ANFIS - Analiza Procesu Uczenia</h3>
        <p style={styles.subtitle}>
          Wizualizacja architektury Adaptive Neuro-Fuzzy Inference System
        </p>
      </div>

      <div style={styles.tabs}>
        <button
          style={{
            ...styles.tab,
            ...(activeTab === 'membership' ? styles.activeTab : {})
          }}
          onClick={() => setActiveTab('membership')}
        >
          📊 Funkcje Przynależności
        </button>
        <button
          style={{
            ...styles.tab,
            ...(activeTab === 'rules' ? styles.activeTab : {})
          }}
          onClick={() => setActiveTab('rules')}
        >
          🔧 Aktywne Reguły
        </button>
        <button
          style={{
            ...styles.tab,
            ...(activeTab === 'learning' ? styles.activeTab : {})
          }}
          onClick={() => setActiveTab('learning')}
        >
          🚀 Proces Uczenia
        </button>
      </div>

      <div style={styles.content}>
        {activeTab === 'membership' && renderMembershipFunctions()}
        {activeTab === 'rules' && renderActiveRules()}
        {activeTab === 'learning' && renderLearning()}
      </div>
    </div>
  );
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
  tabs: {
    display: 'flex',
    gap: '10px',
    marginBottom: '20px',
    borderBottom: '1px solid #363a45',
    paddingBottom: '10px'
  },
  tab: {
    padding: '10px 20px',
    background: '#2a2e39',
    border: 'none',
    borderRadius: '6px 6px 0 0',
    color: '#888',
    cursor: 'pointer',
    fontSize: '0.9em',
    fontWeight: '600',
    transition: 'all 0.2s'
  },
  activeTab: {
    background: '#2962ff',
    color: '#fff'
  },
  content: {
    minHeight: '400px'
  },
  mfContainer: {
    marginBottom: '30px',
    padding: '15px',
    background: '#131722',
    borderRadius: '6px',
    border: '1px solid #2a2e39'
  },
  featureTitle: {
    color: '#d1d4dc',
    marginTop: 0,
    display: 'flex',
    alignItems: 'center',
    gap: '10px'
  },
  directionBadge: {
    fontSize: '0.7em',
    background: '#2a2e39',
    padding: '4px 10px',
    borderRadius: '12px',
    color: '#89b4fa'
  },
  plotRow: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '15px',
    marginBottom: '15px'
  },
  plotBox: {
    background: '#1a1d26',
    padding: '10px',
    borderRadius: '6px'
  },
  plotTitle: {
    margin: '0 0 10px 0',
    color: '#888',
    fontSize: '0.85em',
    textAlign: 'center'
  },
  equationBox: {
    background: '#2a2e39',
    padding: '12px',
    borderRadius: '6px',
    fontSize: '0.85em',
    lineHeight: '1.6',
    color: '#b0b3b8',
    fontFamily: 'monospace'
  },
  rulesContainer: {
    padding: '10px'
  },
  inputControls: {
    background: '#131722',
    padding: '15px',
    borderRadius: '6px',
    marginBottom: '20px'
  },
  sectionTitle: {
    marginTop: 0,
    color: '#d1d4dc',
    fontSize: '1em'
  },
  sliderRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    marginBottom: '10px'
  },
  sliderLabel: {
    width: '120px',
    color: '#888',
    fontSize: '0.9em'
  },
  slider: {
    flex: 1,
    cursor: 'pointer'
  },
  sliderValue: {
    width: '40px',
    textAlign: 'right',
    color: '#d1d4dc',
    fontWeight: 'bold'
  },
  rulesTable: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: '0.85em'
  },
  tableHeader: {
    background: '#2a2e39',
    color: '#888',
    textAlign: 'left'
  },
  tableRow: {
    borderBottom: '1px solid #2a2e39'
  },
  tableCell: {
    padding: '10px',
    color: '#d1d4dc'
  },
  progressBar: {
    position: 'relative',
    width: '100%',
    height: '20px',
    background: '#1a1d26',
    borderRadius: '4px',
    overflow: 'hidden'
  },
  progressFill: {
    height: '100%',
    transition: 'width 0.3s'
  },
  progressText: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    fontSize: '0.75em',
    fontWeight: 'bold',
    color: '#fff',
    textShadow: '0 0 3px rgba(0,0,0,0.8)'
  },
  infoBox: {
    background: '#2a2e39',
    padding: '15px',
    borderRadius: '6px',
    marginTop: '20px',
    fontSize: '0.85em',
    lineHeight: '1.8',
    color: '#b0b3b8',
    borderLeft: '4px solid #2962ff'
  },
  learningContainer: {
    display: 'grid',
    gridTemplateColumns: '1fr',
    gap: '20px'
  },
  emptyState: {
    textAlign: 'center',
    padding: '60px',
    color: '#666',
    fontSize: '1.1em'
  }
};

export default AnfisEducationalPanel;