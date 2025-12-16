import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const InfoBadges = ({ config, onConfigChange, ticker }) => {
  const [generating, setGenerating] = useState(false);

  // 1. GENERATOR AUTO-ML
  const handleAutoStrategy = async () => {
    setGenerating(true);
    try {
      const res = await axios.post('http://127.0.0.1:5000/api/auto_strategy', { ticker });
      onConfigChange(res.data);
      alert("Strategia AI wygenerowana!");
    } catch (err) {
      alert("Błąd generowania strategii.");
    } finally {
      setGenerating(false);
    }
  };

  // 2. STRATEGIA KLASYCZNA (Dostrojona)
  const handleClassicStrategy = () => {
    const classicConfig = {
      'RSI': { 
          enabled: true, 
          weight: 1.0, 
          direction: -1 // RSI 30 (Low) = BUY
      },
      'VIX': { 
          enabled: true, 
          weight: 1.0, 
          direction: 1 // VIX High = Panic = BUY
      },
      'MACD': { 
          enabled: true, 
          weight: 1.5, // Zwiększona waga dla trendu
          direction: 1 // MACD High = Trend Up = BUY
      },
      'Yield_Curve': { 
          enabled: true, 
          weight: 1.0, 
          direction: 1 // Yield Positive = Healthy = BUY
      },
      'M2_Liquidity': { 
          enabled: true, 
          weight: 1.2, // Płynność jest ważna
          direction: 1 // M2 Growth = BUY
      }
    };
    onConfigChange(classicConfig);
  };
  
  const handleChange = (key, field, value) => {
    onConfigChange(prev => ({
      ...prev,
      [key]: { ...prev[key], [field]: value }
    }));
  };

  const features = Object.keys(config);

  const formatLabel = (key) => {
    if (key.includes('ROC')) return `Momentum (${key})`;
    if (key.includes('Volat')) return `Zmienność (${key})`;
    if (key.includes('DistSMA')) return `Trend SMA (${key})`;
    if (key === 'M2_Liquidity') return 'Płynność M2';
    if (key === 'Yield_Curve') return 'Yield Curve';
    return key;
  };

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
        <h4 style={{ margin: 0, fontWeight: 600 }}>
          Aktywne Wskaźniki
        </h4>
        
        <div style={{ display: 'flex', gap: '10px' }}>
            <button 
              onClick={handleClassicStrategy}
              className="ai-button"
              style={{ 
                padding: '8px 15px', borderRadius: '20px', fontSize: '0.85em',
                background: '#444', border: '1px solid #555' 
              }}
              title="RSI, VIX, MACD, Yield, M2"
            >
              🏛️ Klasyczna
            </button>

            <button 
              onClick={handleAutoStrategy} 
              disabled={generating}
              className="ai-button"
              style={{ 
                padding: '8px 15px', borderRadius: '20px', fontSize: '0.85em',
                background: 'linear-gradient(135deg, #00c853 0%, #64dd17 100%)' 
              }}
            >
              {generating ? 'Szukanie...' : '🚀 Generuj (AI)'}
            </button>
        </div>
      </div>

      <div className="indicators-bar">
        {features.length === 0 && <div style={{color:'#666', fontSize:'0.9em', padding:'10px'}}>Wybierz strategię powyżej.</div>}

        {features.map((key) => {
          const itemConfig = config[key];
          const colorHash = key.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
          const colors = ['#cba6f7', '#fab387', '#a6e3a1', '#89b4fa', '#f9e2af', '#f38ba8'];
          const color = colors[colorHash % colors.length];

          return (
            <div 
              key={key} 
              className={`indicator-badge ${!itemConfig.enabled ? 'disabled' : ''}`}
              style={{ borderLeft: `4px solid ${itemConfig.enabled ? color : '#444'}` }}
            >
              <input 
                type="checkbox" 
                checked={itemConfig.enabled} 
                onChange={(e) => handleChange(key, 'enabled', e.target.checked)} 
              />
              
              <div style={{display:'flex', flexDirection:'column'}}>
                <span style={{ color: itemConfig.enabled ? '#fff' : '#888', fontWeight: 600, fontSize: '0.9em' }}>
                  {formatLabel(key)}
                </span>
                <span style={{fontSize: '0.7em', color: '#666'}}>
                   Kierunek: {itemConfig.direction > 0 ? 'Pro (+)' : 'Contra (-)'}
                </span>
              </div>
              
              {itemConfig.enabled && (
                <div className="slider-container">
                  <input 
                    type="range" min="0.0" max="3.0" step="0.1"
                    value={itemConfig.weight} 
                    onChange={(e) => handleChange(key, 'weight', parseFloat(e.target.value))} 
                  />
                  <span className="weight-label">{itemConfig.weight.toFixed(1)}</span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default InfoBadges;