// src/InfoBadges.jsx
import React from 'react';
import './App.css';

const InfoBadges = ({ config, onConfigChange }) => {
  
  const handleChange = (key, field, value) => {
    // Aktualizujemy stan konfiguracji w App.jsx
    onConfigChange(prev => ({
      ...prev,
      [key]: {
        ...prev[key],
        [field]: value
      }
    }));
  };

  // Definicja wskaźników (musi pasować do kluczy w stanie)
  const badges = [
    { key: 'rsi', label: 'RSI (Technika)', color: '#cba6f7' },
    { key: 'vix', label: 'VIX (Emocje)', color: '#fab387' },
    { key: 'yield', label: 'Yield (Makro)', color: '#a6e3a1' }
  ];

  return (
    <div className="indicators-bar">
      {badges.map(({ key, label, color }) => {
        const itemConfig = config[key]; // Pobieramy aktualny stan (enabled, weight)
        
        return (
          <div 
            key={key} 
            className={`indicator-badge control-badge ${!itemConfig.enabled ? 'disabled' : ''}`}
            style={{ borderLeft: `3px solid ${itemConfig.enabled ? color : '#555'}` }}
          >
            {/* 1. Checkbox (Włącz/Wyłącz) */}
            <input 
              type="checkbox" 
              checked={itemConfig.enabled}
              onChange={(e) => handleChange(key, 'enabled', e.target.checked)}
              style={{ marginRight: '8px', cursor: 'pointer' }}
            />

            {/* 2. Nazwa */}
            <span style={{ color: itemConfig.enabled ? color : '#888', fontWeight: 'bold' }}>
              {label}
            </span>

            {/* 3. Suwak Wagi (Widoczny tylko gdy włączony) */}
            {itemConfig.enabled && (
              <div className="slider-container">
                <input 
                  type="range" 
                  min="0.1" max="2.0" step="0.1"
                  value={itemConfig.weight}
                  onChange={(e) => handleChange(key, 'weight', parseFloat(e.target.value))}
                  title={`Waga: ${itemConfig.weight}`}
                />
                <span className="weight-label">{itemConfig.weight.toFixed(1)}x</span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default InfoBadges;