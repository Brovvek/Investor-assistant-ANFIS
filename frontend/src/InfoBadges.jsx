// frontend/src/InfoBadges.jsx
import React, { useState } from 'react';
import './App.css'; // Upewnij się, że importujesz CSS

const InfoBadges = ({ config, onConfigChange, ticker }) => {
  const [optimizing, setOptimizing] = useState(false);
  const [progress, setProgress] = useState(0);

  // Dane opisowe dla tooltipów
  const infoData = {
    rsi: {
      fullTitle: 'RSI (Relative Strength Index)',
      desc: 'Mierzy dynamikę ceny. Wykrywa momenty wyprzedania (dołki) i wykupienia (szczyty).',
      high: '▲ >70: Drogo (Sygnał Sprzedaży)',
      low: '▼ <30: Tanio (Sygnał Kupna)'
    },
    vix: {
      fullTitle: 'VIX (Indeks Strachu)',
      desc: 'Oczekiwana zmienność rynku. Działa odwrotnie do giełdy.',
      high: '▲ Wysoki: Panika (Często okazja do kupna)',
      low: '▼ Niski: Chciwość (Ryzyko spadków)'
    },
    yield: {
      fullTitle: 'Yield Curve (10Y-2Y)',
      desc: 'Różnica oprocentowania obligacji. Najlepszy predyktor recesji.',
      high: '▲ Dodatnia: Zdrowa gospodarka',
      low: '▼ Ujemna: Inwersja (Recesja)'
    },
    macd: {
      fullTitle: 'MACD (Trend)',
      desc: 'Śledzi siłę i kierunek trendu.',
      high: '▲ Wysoki: Silny trend wzrostowy',
      low: '▼ Niski: Trend spadkowy'
    },
    m2: {
      fullTitle: 'M2 Money Supply',
      desc: 'Podaż pieniądza (Płynność). Paliwo dla wzrostów giełdowych.',
      high: '▲ Rośnie: Banki drukują -> Akcje rosną',
      low: '▼ Spada: Mniej pieniądza -> Spadki'
    }
  };

  const handleOptimize = async () => {
    setOptimizing(true);
    setProgress(0);

    try {
      const response = await fetch('http://127.0.0.1:5000/api/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            if (data.status === 'progress') setProgress(data.value);
            else if (data.status === 'done') applyWeights(data.result);
            else if (data.status === 'error') alert("Błąd AI: " + data.message);
          } catch (e) { console.error(e); }
        }
      }
    } catch (err) {
      alert("Błąd połączenia.");
    } finally {
      setOptimizing(false);
      setProgress(100);
      setTimeout(() => setProgress(0), 2000);
    }
  };

  const applyWeights = (bestWeights) => {
    const newConfig = { ...config };
    Object.keys(bestWeights).forEach(key => {
      if (!newConfig[key]) newConfig[key] = { enabled: true, weight: 1.0 };
      newConfig[key].weight = bestWeights[key];
      newConfig[key].enabled = bestWeights[key] > 0.1;
    });
    onConfigChange(newConfig);
  };
  
  const handleChange = (key, field, value) => {
    onConfigChange(prev => ({
      ...prev,
      [key]: { ...prev[key], [field]: value }
    }));
  };

  const badges = [
    { key: 'rsi', label: 'RSI (Technika)', color: '#cba6f7' },
    { key: 'vix', label: 'VIX (Strach)', color: '#fab387' },
    { key: 'yield', label: 'Yield (Makro)', color: '#a6e3a1' },
    { key: 'macd', label: 'MACD (Trend)', color: '#89b4fa' },
    { key: 'm2', label: 'M2 (Płynność)', color: '#f9e2af' }
  ];

  return (
    <div>
      {/* NAGŁÓWEK SEKCI KONFIGURACJI */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
        <h4 style={{ margin: 0, fontWeight: 600 }}>
          Konfiguracja Wag 
          <span style={{fontSize:'0.8em', fontWeight:'normal', color:'#666', marginLeft: '10px'}}>
            (Ręcznie lub AI)
          </span>
        </h4>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          {optimizing && (
            <div style={{ width: '120px', height: '8px', background: '#333', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{ width: `${progress}%`, height: '100%', background: '#2962ff', transition: 'width 0.3s' }}></div>
            </div>
          )}
          
          <button 
            onClick={handleOptimize} 
            disabled={optimizing}
            className="ai-button"
            style={{ padding: '8px 16px', borderRadius: '20px', fontSize: '0.85em' }}
          >
            {optimizing ? `Trenowanie... ${progress}%` : '✨ Auto-Tune AI'}
          </button>
        </div>
      </div>

      {/* PASEK KAFELKÓW (INDICATORS BAR) */}
      <div className="indicators-bar">
        {badges.map(({ key, label, color }) => {
          const itemConfig = config[key] || { enabled: false, weight: 1.0 };
          const info = infoData[key] || { fullTitle: label };

          return (
            <div 
              key={key} 
              className={`indicator-badge ${!itemConfig.enabled ? 'disabled' : ''}`}
              style={{ borderLeft: `4px solid ${itemConfig.enabled ? color : '#444'}` }}
            >
              {/* Checkbox */}
              <input 
                type="checkbox" 
                checked={itemConfig.enabled} 
                onChange={(e) => handleChange(key, 'enabled', e.target.checked)} 
              />
              
              {/* Nazwa */}
              <span style={{ color: itemConfig.enabled ? '#fff' : '#888', fontWeight: 600, cursor: 'help' }}>
                {label}
              </span>
              
              {/* Suwak (widoczny tylko gdy włączony) */}
              {itemConfig.enabled && (
                <div className="slider-container">
                  <input 
                    type="range" 
                    min="0.0" max="2.0" step="0.1" 
                    value={itemConfig.weight} 
                    onChange={(e) => handleChange(key, 'weight', parseFloat(e.target.value))} 
                  />
                  <span className="weight-label">{itemConfig.weight.toFixed(1)}</span>
                </div>
              )}

              {/* TOOLTIP (Niewidoczny, pojawia się po najechaniu CSS-em) */}
              <div className="custom-tooltip">
                <div className="tooltip-title" style={{ color: color }}>{info.fullTitle}</div>
                <div className="tooltip-desc">{info.desc}</div>
                <div className="tooltip-val" style={{ color: '#00e676' }}>{info.high}</div>
                <div className="tooltip-val" style={{ color: '#ff5252' }}>{info.low}</div>
              </div>

            </div>
          );
        })}
      </div>
    </div>
  );
};

export default InfoBadges;