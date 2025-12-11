import React from 'react';
import './App.css';

const indicators = [
  {
    key: 'anfis',
    name: 'ANFIS (AI)',
    color: '#f38ba8',
    fullTitle: 'Oscylator Nastrojów (ANFIS)',
    desc: 'Syntetyczna ocena rynku oparta na sztucznej inteligencji. Łączy technikę, emocje i makro.',
    high: '▲ Wysoki: Okazja (Panika/Wyprzedanie)',
    low: '▼ Niski: Ryzyko (Euforia/Bańka)'
  },
  {
    key: 'rsi',
    name: 'RSI',
    color: '#cba6f7',
    fullTitle: 'Relative Strength Index',
    desc: 'Wskaźnik techniczny mierzący dynamikę ceny (prędkość zmian) w ostatnich 14 dniach.',
    high: '▲ >70: Wykupienie (Drogo)',
    low: '▼ <30: Wyprzedanie (Tanio)'
  },
  {
    key: 'vix',
    name: 'VIX (Strach)',
    color: '#fab387',
    fullTitle: 'Volatility Index (Emocje)',
    desc: 'Poziom strachu inwestorów. Ranking historyczny z ostatnich 2 lat.',
    high: '▲ Wysoki: Panika (Często dno ceny)',
    low: '▼ Niski: Zbyt duży spokój (Ryzyko)'
  },
  {
    key: 'yield',
    name: 'Yield Curve',
    color: '#a6e3a1',
    fullTitle: 'Krzywa Dochodowości (10Y-2Y)',
    desc: 'Najważniejszy wskaźnik makroekonomiczny ostrzegający przed recesją.',
    high: '▲ Dodatnia: Zdrowa gospodarka',
    low: '▼ Ujemna: Ryzyko recesji (Inwersja)'
  }
];

const InfoBadges = () => {
  return (
    <div className="indicators-bar">
      {indicators.map((item) => (
        <div key={item.key} className="indicator-badge" style={{ borderLeft: `3px solid ${item.color}` }}>
          <span style={{ color: item.color }}>{item.name}</span>
          <span className="info-icon">ⓘ</span>
          
          {/* TOOLTIP HTML */}
          <div className="custom-tooltip">
            <div className="tooltip-title" style={{ color: item.color }}>{item.fullTitle}</div>
            <div className="tooltip-desc">{item.desc}</div>
            <div className="tooltip-val" style={{ color: '#66bb6a' }}>{item.high}</div>
            <div className="tooltip-val" style={{ color: '#ef5350' }}>{item.low}</div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default InfoBadges;