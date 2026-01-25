import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import './App.css';

const fmt = (value, decimals = 2) => {
  if (value === undefined || value === null || isNaN(value)) return '—';
  return Number(value).toFixed(decimals);
};

const AnfisMLPanel = ({ ticker, config }) => {
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
  const [trainingDays, setTrainingDays] = useState(0);
  
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [liveMetrics, setLiveMetrics] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [rawResponse, setRawResponse] = useState('');
  const [activeTab, setActiveTab] = useState('predictions');
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [saveNotes, setSaveNotes] = useState('');
  const [selectedIds, setSelectedIds] = useState(new Set());
  const [expandedIds, setExpandedIds] = useState(new Set());
  const [legendExpanded, setLegendExpanded] = useState(false);

  useEffect(() => { fetchHistory(); }, []);

  const fetchHistory = async () => {
    setHistoryLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:5000/api/anfis_ml/get_history');
      const data = await response.json();
      setTrainingHistory(data.history || []);
      setSelectedIds(new Set());
      setExpandedIds(new Set());
    } catch (err) {
      setTrainingHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  const toggleSelectRecord = (id, e) => {
    if (e) e.stopPropagation();
    setSelectedIds(prev => {
      const newSet = new Set(prev);
      newSet.has(id) ? newSet.delete(id) : newSet.add(id);
      return newSet;
    });
  };

  const toggleSelectAll = () => {
    setSelectedIds(selectedIds.size === trainingHistory.length ? new Set() : new Set(trainingHistory.map(r => r.id)));
  };

  const toggleExpandRecord = (id) => {
    setExpandedIds(prev => {
      const newSet = new Set(prev);
      newSet.has(id) ? newSet.delete(id) : newSet.add(id);
      return newSet;
    });
  };

  const toggleExpandAll = () => {
    setExpandedIds(expandedIds.size === trainingHistory.length ? new Set() : new Set(trainingHistory.map(r => r.id)));
  };

  const deleteSelected = async () => {
    const ids = selectedIds.size > 0 ? Array.from(selectedIds) : [];
    if (!window.confirm(ids.length > 0 ? `Usunąć ${ids.length} rekordów?` : 'Usunąć WSZYSTKIE rekordy?')) return;
    try {
      await fetch('http://127.0.0.1:5000/api/anfis_ml/delete_history', {
        method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ids })
      });
      fetchHistory();
    } catch (err) { alert('Błąd: ' + err.message); }
  };

  const exportSelected = async () => {
    const ids = selectedIds.size > 0 ? Array.from(selectedIds) : [];
    try {
      const response = await fetch('http://127.0.0.1:5000/api/anfis_ml/export_selected', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ids })
      });
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `anfis_history_${new Date().toISOString().slice(0,10)}.csv`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err) { alert('Błąd: ' + err.message); }
  };

  const saveResults = async () => {
    if (!results?.metrics) return alert('Brak wyników');
    try {
      const response = await fetch('http://127.0.0.1:5000/api/anfis_ml/save_results', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker, prediction_type: predictionType, epochs: parseInt(epochs),
          num_mfs: parseInt(numMfs), mf_type: mfType, optimizer,
          learning_rate: parseFloat(learningRate), batch_size: parseInt(batchSize),
          lookahead: parseInt(lookahead), scaler_type: scalerType, hybrid,
          training_days: parseInt(trainingDays) || 0,
          ...results.metrics,
          features: Object.keys(config || {}).filter(k => config[k]?.enabled).join(', '),
          notes: saveNotes
        })
      });
      const result = await response.json();
      if (result.success) { alert('Zapisano!'); setSaveNotes(''); fetchHistory(); }
    } catch (err) { alert('Błąd: ' + err.message); }
  };

  const startTraining = async () => {
    setIsTraining(true); setProgress(0); setResults(null); setError('');
    try {
      const response = await fetch('http://127.0.0.1:5000/api/anfis_ml/train_stream', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker, config, epochs: parseInt(epochs), num_mfs: parseInt(numMfs),
          batch_size: parseInt(batchSize), learning_rate: parseFloat(learningRate),
          mf_type: mfType, hybrid, optimizer, lookahead: parseInt(lookahead),
          prediction_type: predictionType, scaler_type: scalerType,
          early_stopping_patience: parseInt(earlyStoppingPatience),
          training_days: parseInt(trainingDays) || 0
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
                setProgress(data.progress || 0);
                setCurrentEpoch(data.epoch || 0);
                setLiveMetrics({ val_rmse: data.val_rmse, direction_acc: data.direction_acc });
              } else if (data.status === 'done') {
                setRawResponse(JSON.stringify(data, null, 2).slice(0, 2000));
                setResults(data); setProgress(100);
              } else if (data.status === 'error') setError(data.message);
            } catch (e) {}
          }
        }
      }
    } catch (err) { setError(err.message); }
    finally { setIsTraining(false); }
  };

  const canTrain = Object.values(config || {}).some(v => v?.enabled);
  const getMetric = (key, fb = 0) => results?.metrics?.[key] ?? results?.metrics?.[`test_${key}`] ?? fb;
  const getPredictions = () => results?.predictions || { dates: [], actual: [], predicted: [] };
  const getTrainHistory = () => results?.training_history || { epochs: [], train_loss: [], val_loss: [] };
  const getMFs = () => results?.membership_functions || {};
  const getRules = () => results?.rules || [];
  const getImportance = () => results?.feature_importance || {};

  const getAccuracyClass = (acc) => acc > 55 ? 'green' : acc > 50 ? '' : 'red';

  const HistoryRecord = ({ row }) => {
    const isExpanded = expandedIds.has(row.id);
    const isSelected = selectedIds.has(row.id);
    const accClass = getAccuracyClass(row.direction_accuracy || 0);
    
    return (
      <div className={`history-record ${isSelected ? 'selected' : ''}`}>
        <div className={`history-record-header ${isExpanded ? 'expanded' : ''}`} onClick={() => toggleExpandRecord(row.id)}>
          <div onClick={e => e.stopPropagation()} className="mr-md">
            <input type="checkbox" checked={isSelected} onChange={e => toggleSelectRecord(row.id, e)} />
          </div>
          <span className={`expand-icon ${isExpanded ? 'expanded' : ''}`}>▶</span>
          <span className="history-record-id">{row.id || '—'}</span>
          <span className="history-record-ticker">{row.ticker || '—'}</span>
          <span className="history-record-date">{row.timestamp?.slice(0, 16) || '—'}</span>
          <span className={`history-record-accuracy text-${accClass || 'primary'}`}>Celność: {fmt(row.direction_accuracy, 1)}%</span>
          <span className={`history-record-correlation text-${(row.correlation||0)*100 > 50 ? 'green' : (row.correlation||0)*100 > 30 ? 'yellow' : 'orange'}`}>Korelacja: {fmt((row.correlation||0)*100, 1)}%</span>
          <span className="history-record-type">{row.prediction_type || '—'}</span>
          <span className="history-record-notes">{row.notes || ''}</span>
        </div>
        {isExpanded && (
          <div className="history-record-content">
            <div className="history-section">
              <h4 className="history-section-title">📊 Metryki</h4>
              <div className="grid-metrics">
                {[['Celność', row.direction_accuracy, accClass], ['RMSE', row.rmse, 'blue'], 
                  ['MAPE', row.mape, 'orange'], ['R²', (row.r_squared||0)*100, 'green-alt'],
                  ['Korelacja', (row.correlation||0)*100, 'purple']
                ].map(([label, val, colorClass]) => (
                  <div key={label} className="stat-card">
                    <div className="stat-card-label">{label}</div>
                    <div className={`stat-card-value ${colorClass}`}>{fmt(val, label === 'RMSE' ? 4 : 2)}{label !== 'RMSE' ? '%' : ''}</div>
                  </div>
                ))}
              </div>
            </div>
            <div className="history-section">
              <h4 className="history-section-title">⚙️ Parametry</h4>
              <div className="grid-params">
                {[['Typ', row.prediction_type], ['Epoki', row.epochs], ['MF', row.num_mfs], ['Typ MF', row.mf_type],
                  ['Optimizer', row.optimizer], ['LR', row.learning_rate], ['Batch', row.batch_size], ['Horyzont', row.lookahead],
                  ['Scaler', row.scaler_type], ['Hybrid', row.hybrid ? 'Tak' : 'Nie'], ['Dni', row.training_days || 'Wszystkie'], ['Reguły', row.num_rules]
                ].map(([k, v]) => (
                  <div key={k} className="param-row"><span className="param-key">{k}:</span><span className="param-value">{v ?? '—'}</span></div>
                ))}
              </div>
            </div>
            <div className="history-section">
              <h4 className="history-section-title">🔧 Cechy</h4>
              <div className="info-box-features">{row.features || 'Brak'}</div>
            </div>
            {row.notes && <div className="history-section">
              <h4 className="history-section-title">📝 Notatki</h4>
              <div className="info-box-features italic">{row.notes}</div>
            </div>}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="p-sm">
      <div className="legend-section">
        <div className="legend-header" onClick={() => setLegendExpanded(!legendExpanded)}>
          <span className="legend-title">📚 Legenda Metryk i Parametrów</span>
          <span className={`legend-icon ${legendExpanded ? 'expanded' : ''}`}>▼</span>
        </div>
        {legendExpanded && (
          <div className="legend-content">
            <div className="legend-grid">
              <div className="legend-item">
                <div className="legend-item-title">🎯 Celność Kierunku (Direction Accuracy)</div>
                <div className="legend-item-desc">Procent poprawnych prognoz kierunku ruchu ceny (wzrost/spadek). <span className="text-green">Powyżej 55%</span> oznacza użyteczny model. <span className="text-orange">Powyżej 50%</span> jest marginalne.</div>
              </div>
              <div className="legend-item">
                <div className="legend-item-title">📉 RMSE (Root Mean Square Error)</div>
                <div className="legend-item-desc">Średni błąd kwadratowy predykcji (w jednostkach ceny). Niższe wartości są lepsze. <span className="text-green">&lt;0.05</span> = doskonały, <span className="text-yellow">&lt;0.1</span> = dobry.</div>
              </div>
              <div className="legend-item">
                <div className="legend-item-title">📊 MAPE (Mean Absolute Percentage Error)</div>
                <div className="legend-item-desc">Średni błąd procentowy predykcji. <span className="text-green">&lt;5%</span> = doskonały, <span className="text-yellow">&lt;10%</span> = dobry, <span className="text-orange">&gt;10%</span> = słaby.</div>
              </div>
              <div className="legend-item">
                <div className="legend-item-title">📈 R² (Coefficient of Determination)</div>
                <div className="legend-item-desc">Udział wariancji wyjaśnionej przez model (0-1). <span className="text-green">&gt;0.2</span> = użyteczny, <span className="text-yellow">&gt;0.1</span> = słaby. Im wyżej, tym lepiej.</div>
              </div>
              <div className="legend-item">
                <div className="legend-item-title">🔗 Korelacja (Correlation)</div>
                <div className="legend-item-desc">Pearson correlation między prognozą a rzeczywistymi wartościami (-1 do 1). <span className="text-green">&gt;0.5</span> = silna, <span className="text-yellow">&gt;0.3</span> = słaba.</div>
              </div>
              <div className="legend-item">
                <div className="legend-item-title">🔀 Train/Test</div>
                <div className="legend-item-desc">Rozmiar zbiorów treningowego i testowego (liczba próbek). Większy zbiór treningowy = lepsze dopasowanie, ale może być overfitting.</div>
              </div>
              <div className="legend-item">
                <div className="legend-item-title">⚙️ Epoki</div>
                <div className="legend-item-desc">Liczba przejść przez całą próbkę treningową. Więcej epok = lepsze dopasowanie, ale wolniej i ryzyko overfittingu.</div>
              </div>
              <div className="legend-item">
                <div className="legend-item-title">🌊 Liczba MF (Membership Functions)</div>
                <div className="legend-item-desc">Liczba funkcji przynależności w każdej zmiennej ANFIS. Więcej = bardziej elastyczny model, ale bardziej złożony.</div>
              </div>
              <div className="legend-item">
                <div className="legend-item-title">🎨 Typ MF</div>
                <div className="legend-item-desc"><span className="text-blue">Gaussowska</span> = gładka, <span className="text-blue">Dzwonowa</span> = asymetryczna, <span className="text-blue">Trójkątna</span> = ostra. Zwykle Gaussowska daje najlepsze wyniki.</div>
              </div>
              <div className="legend-item">
                <div className="legend-item-title">⚡ Optymalizator</div>
                <div className="legend-item-desc"><span className="text-blue">Adam</span> = domyślny, najszybszy; <span className="text-blue">AdamW</span> = z regularyzacją; <span className="text-blue">SGD</span> = wolny ale niezawodny.</div>
              </div>
              <div className="legend-item">
                <div className="legend-item-title">🔄 Scaler</div>
                <div className="legend-item-desc">Normalizacja danych wejściowych: <span className="text-blue">Robust</span> = odporny na outliers (mediana/IQR), <span className="text-blue">Standard</span> = klasyczna normalizacja (średnia/std), <span className="text-blue">MinMax</span> = skalowanie do [0,1].</div>
              </div>
              <div className="legend-item">
                <div className="legend-item-title">📍 Scatter (Wykres rozrzutu)</div>
                <div className="legend-item-desc">Porównanie prognoz i rzeczywistych wartości punkt po punkcie. <span className="text-green">Zielone punkty</span> = prawidłowy kierunek, <span className="text-red">czerwone</span> = błędny kierunek. Idealnie mają leżeć na linii y=x.</div>
              </div>
            </div>
          </div>
        )}
      </div>
      
      <div className="panel-dark grid-auto mb-lg">
        {[['Typ Predykcji', predictionType, setPredictionType, [['returns','% Zmiana'],['log_returns','Log Returns'],['direction','Kierunek'],['price','Surowa Cena']]],
          ['Typ MF', mfType, setMfType, [['gauss','Gaussowska'],['bell','Dzwonowa'],['tri','Trójkątna']]],
          ['Optymalizator', optimizer, setOptimizer, [['adam','Adam'],['adamw','AdamW'],['sgd','SGD']]],
          ['Batch Size', batchSize, setBatchSize, [['32','32'],['64','64'],['128','128']]],
          ['Scaler', scalerType, setScalerType, [['robust','Robust'],['standard','Standard'],['minmax','MinMax']]],
          ['Dni treningowe', trainingDays, setTrainingDays, [['0','Wszystkie'],['252','1 rok'],['504','2 lata'],['1260','5 lat']]]
        ].map(([label, val, setter, opts]) => (
          <div key={label}><label className="label">{label}</label>
            <select value={val} onChange={e => setter(e.target.value)} className="input-full">
              {opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </div>
        ))}
        {[['Epoki', epochs, setEpochs, 10, 500], ['Liczba MF', numMfs, setNumMfs, 2, 7], 
          ['Learning Rate', learningRate, setLearningRate, 0.001, 0.1], ['Horyzont', lookahead, setLookahead, 1, 30],
          ['Early Stop', earlyStoppingPatience, setEarlyStoppingPatience, 5, 50]
        ].map(([label, val, setter, min, max]) => (
          <div key={label}><label className="label">{label}</label>
            <input type="number" value={val} onChange={e => setter(e.target.value)} min={min} max={max} step={label === 'Learning Rate' ? 0.001 : 1} className="input-full" />
          </div>
        ))}
        <div className="flex-row pt-lg">
          <label><input type="checkbox" checked={hybrid} onChange={e => setHybrid(e.target.checked)} className="mr-sm" />Hybrid (LSE)</label>
        </div>
      </div>

      <div className="flex-row flex-wrap gap-xl mb-lg">
        <button onClick={startTraining} disabled={isTraining || !canTrain} className={`btn-lg ${canTrain ? 'btn-gradient' : ''}`}>
          {isTraining ? `Trenuję... ${progress}%` : 'Rozpocznij Uczenie ANFIS'}
        </button>
        {!canTrain && <span className="text-red">Wybierz min. 1 wskaźnik</span>}
        {isTraining && liveMetrics && <div className="info-box info-box-compact">Epoka: <span className="text-blue">{currentEpoch}/{epochs}</span> | RMSE: <span className="text-orange">{fmt(liveMetrics.val_rmse, 4)}</span> | Dir: <span className={`text-${liveMetrics.direction_acc > 55 ? 'green' : liveMetrics.direction_acc > 50 ? 'yellow' : 'red'}`}>{fmt(liveMetrics.direction_acc, 1)}%</span></div>}
      </div>

      {isTraining && <div className="progress-bar"><div className="progress-bar-fill" style={{ width: `${progress}%` }} /></div>}
      {error && <div className="alert-error">{error}</div>}

      {results && (
        <div>
          <div className="stats-grid-sm mb-lg">
            {[['Celność Kierunku', 'direction_accuracy', 50, v => v > 55],['RMSE', 'rmse', 0],['MAPE', 'mape', 0],['R²', 'r_squared', 0, v => v > 0.2],['Korelacja', 'correlation', 0],['Train/Test', null]
            ].map(([label, key, fb, hl]) => {
              const value = getMetric(key, fb);
              const getColorClass = (k, v) => {
                if (!k) return '';
                if (k === 'direction_accuracy') return v > 55 ? 'green' : v > 50 ? 'yellow' : 'red';
                if (k === 'rmse') return v < 0.05 ? 'green' : v < 0.1 ? 'yellow' : 'orange';
                if (k === 'mape') return v < 5 ? 'green' : v < 10 ? 'yellow' : 'orange';
                if (k === 'r_squared') return v * 100 > 20 ? 'green' : v * 100 > 10 ? 'yellow' : 'red';
                if (k === 'correlation') return v * 100 > 50 ? 'green' : v * 100 > 30 ? 'yellow' : 'orange';
                return '';
              };
              const colorClass = getColorClass(key, value);
              return (
                <div key={label} className={`stat-box ${hl && hl(value) ? 'highlight' : ''}`}>
                  <div className="stat-label">{label}</div>
                  <div className={`stat-val text-${colorClass}`}>
                    {key ? (key === 'r_squared' || key === 'correlation' ? fmt(value * 100, 1) : fmt(value, key === 'rmse' ? 4 : 1)) + (key !== 'rmse' ? '%' : '') : `${getMetric('train_samples', '?')} / ${getMetric('test_samples', '?')}`}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="panel-dark mb-lg">
            <div className="flex-row flex-wrap gap-lg">
              <span className="font-bold text-primary">Zapisz:</span>
              <input type="text" placeholder="Notatki..." value={saveNotes} onChange={e => setSaveNotes(e.target.value)} className="input-full input-flex" />
              <button onClick={saveResults} className="btn-success">Zapisz do CSV</button>
            </div>
          </div>

          <div className="tabs-container">
            {['predictions','scatter','loss','mf','importance','rules','debug'].map(tab => (
              <button key={tab} onClick={() => setActiveTab(tab)} className={`tab-btn ${activeTab === tab ? 'active' : ''}`}>
                {{predictions:'Predykcje',scatter:'Scatter',loss:'Loss',mf:'MF',importance:'Ważność',rules:'Reguły',debug:'Debug'}[tab]}
              </button>
            ))}
          </div>

          <div className="section-box">
            {activeTab === 'predictions' && (getPredictions().dates?.length > 0 ? (
              <Plot data={[
                { x: getPredictions().dates, y: getPredictions().actual, type: 'scatter', mode: 'lines', name: 'Rzeczywiste', line: { color: '#2962ff', width: 2 } },
                { x: getPredictions().dates, y: getPredictions().predicted, type: 'scatter', mode: 'lines', name: 'Predykcja', line: { color: '#00e676', width: 2, dash: 'dot' } }
              ]} layout={{ paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#d1d4dc' }, margin: { t: 40, b: 40, l: 60, r: 20 }, xaxis: { showgrid: false }, yaxis: { gridcolor: '#2a2e39' }, legend: { orientation: 'h', y: 1.1 } }} useResizeHandler className="chart-full" />
            ) : <div className="empty-state">Brak danych</div>)}

            {activeTab === 'scatter' && (getPredictions().actual?.length > 0 ? (() => {
              const preds = getPredictions();
              const allVals = [...preds.actual, ...preds.predicted].filter(v => !isNaN(v));
              const minVal = Math.min(...allVals);
              const maxVal = Math.max(...allVals);
              const diagLine = [minVal, maxVal];
              return (
                <Plot data={[
                  { x: preds.actual, y: preds.predicted, type: 'scatter', mode: 'markers', marker: { color: preds.actual.map((a, i) => Math.sign(a) === Math.sign(preds.predicted[i]) ? '#00e676' : '#ff5252'), size: 5 }, name: 'Predykcje' },
                  { x: diagLine, y: diagLine, type: 'scatter', mode: 'lines', name: 'y=x (Idealna)', line: { color: '#cba6f7', width: 2, dash: 'dash' } }
                ]} layout={{ paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#d1d4dc' }, margin: { t: 40, b: 50, l: 60, r: 20 }, xaxis: { title: 'Rzeczywiste', gridcolor: '#2a2e39' }, yaxis: { title: 'Predykcja', gridcolor: '#2a2e39' }, legend: { orientation: 'h', y: 1.05 } }} useResizeHandler className="chart-full" />
              );
            })() : <div className="empty-state">Brak danych</div>)}

            {activeTab === 'loss' && (getTrainHistory().epochs?.length > 0 ? (
              <div className="grid-charts">
                <Plot data={[{ x: getTrainHistory().epochs, y: getTrainHistory().train_loss, name: 'Train', line: { color: '#2962ff' } }, { x: getTrainHistory().epochs, y: getTrainHistory().val_loss, name: 'Val', line: { color: '#ff5252' } }]} layout={{ title: { text: 'Loss', font: { color: '#d1d4dc', size: 12 } }, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#d1d4dc' }, margin: { t: 35, b: 35, l: 45, r: 10 }, legend: { orientation: 'h', y: 1.15 } }} className="chart-md" />
                <Plot data={[{ x: getTrainHistory().epochs, y: getTrainHistory().val_direction_acc || [], name: 'Celność', line: { color: '#00e676' }, fill: 'tozeroy' }]} layout={{ title: { text: 'Celność', font: { color: '#d1d4dc', size: 12 } }, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#d1d4dc' }, margin: { t: 35, b: 35, l: 45, r: 10 }, yaxis: { range: [40, 70] } }} className="chart-md" />
              </div>
            ) : <div className="empty-state">Brak danych</div>)}

            {activeTab === 'mf' && (Object.keys(getMFs()).length > 0 ? (
              <div className="grid-mf">
                {Object.entries(getMFs()).map(([f, d]) => {
                  const mfsData = d.mfs && typeof d.mfs === 'object' && !Array.isArray(d.mfs)
                    ? Object.entries(d.mfs).map(([mfName, yValues]) => ({ x: d.x, y: yValues, name: mfName, type: 'scatter', mode: 'lines' }))
                    : (d.mfs || []).map((m, i) => ({ x: d.x, y: m.y || m, name: `MF${i+1}`, type: 'scatter', mode: 'lines' }));
                  return <Plot key={f} data={mfsData} layout={{ title: { text: f, font: { color: '#d1d4dc', size: 11 } }, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#888' }, margin: { t: 30, b: 25, l: 35, r: 10 }, showlegend: true, legend: { font: { size: 9 } }, xaxis: { gridcolor: '#2a2e39' }, yaxis: { gridcolor: '#2a2e39', range: [0, 1.05] } }} className="chart-sm" useResizeHandler />;
                })}
              </div>
            ) : <div className="empty-state">Brak danych MF</div>)}

            {activeTab === 'importance' && (Object.keys(getImportance()).length > 0 ? (
              <Plot data={[{ x: Object.values(getImportance()), y: Object.keys(getImportance()), type: 'bar', orientation: 'h', marker: { color: '#2962ff' } }]} layout={{ paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#d1d4dc' }, margin: { t: 40, b: 40, l: 150, r: 20 } }} useResizeHandler className="chart-full" />
            ) : <div className="empty-state">Brak danych</div>)}

                        {activeTab === 'rules' && (
              getRules().length > 0 ? (
                <div className="rules-table-wrapper">
                  <table className="rules-table">
                    <thead>
                      <tr className="rules-table-header"><th>#</th><th>Reguła</th></tr>
                    </thead>
                    <tbody>
                      {getRules().map((rule, i) => (
                        <tr key={i}>
                          <td className="rules-table-number">{i + 1}</td>
                          <td className="rules-table-cell">
                            <span className="rules-if-keyword">IF </span>
                            <span className="rules-antecedent">{rule.antecedent || 'N/A'}</span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : <div className="rules-empty-state">Brak reguł</div>
            )}

            {activeTab === 'debug' && <pre className="debug-pre">{rawResponse || 'Brak'}</pre>}
          </div>
        </div>
      )}

      <div className="mt-lg">
        <div className="flex-between mb-md">
          <h3 className="text-primary m-0">📜 Historia ({trainingHistory.length})</h3>
          <div className="flex-row gap-lg">
            <button onClick={toggleExpandAll} className="btn-secondary btn-pill">{expandedIds.size === trainingHistory.length ? 'Zwiń' : 'Rozwiń'}</button>
            <button onClick={fetchHistory} className="btn-secondary btn-pill">Odśwież</button>
            <button onClick={exportSelected} className="btn-secondary btn-pill">Eksport {selectedIds.size > 0 ? `(${selectedIds.size})` : ''}</button>
            <button onClick={deleteSelected} className="btn-pill btn-danger">Usuń {selectedIds.size > 0 ? `(${selectedIds.size})` : 'Wszystko'}</button>
          </div>
        </div>
        {historyLoading ? <div className="loading-container"><div className="loading-spinner"></div><p className="mt-md">Ładowanie...</p></div> :
         trainingHistory.length === 0 ? <div className="empty-state panel-dark">Brak zapisanych treningów</div> : (
          <div>
            <div className="flex-row gap-md mb-md p-sm select-all-bar">
              <input type="checkbox" checked={selectedIds.size === trainingHistory.length && trainingHistory.length > 0} onChange={toggleSelectAll} />
              <span className="label-sm">Zaznacz wszystko ({selectedIds.size}/{trainingHistory.length})</span>
            </div>
            {trainingHistory.map((row, i) => <HistoryRecord key={row.id || i} row={row} />)}
          </div>
        )}
      </div>
    </div>
  );
};

export default AnfisMLPanel;
