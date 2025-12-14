import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';

const CorrelationPanel = ({ ticker }) => {
  const [corrData, setCorrData] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchCorrelations = async () => {
    setLoading(true);
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/correlations', { ticker });
      setCorrData(response.data);
    } catch (err) { console.error(err); } finally { setLoading(false); }
  };

  useEffect(() => { if (ticker) fetchCorrelations(); }, [ticker]);
  if (!corrData) return null;

  const xValues = Object.keys(corrData);
  const yValues = Object.values(corrData);
  const colors = yValues.map(v => v >= 0 ? '#66bb6a' : '#ef5350');

  return (
    <div style={{ width: '100%', height: '300px' }}>
      <Plot
        data={[{ x: xValues, y: yValues, type: 'bar', marker: { color: colors } }]}
        layout={{
          paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#d1d4dc' },
          yaxis: { title: 'Korelacja (-1 do 1)', range: [-1, 1] }, margin: { t: 10, b: 30, l: 50, r: 20 }
        }}
        useResizeHandler={true} style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
};

export default CorrelationPanel;