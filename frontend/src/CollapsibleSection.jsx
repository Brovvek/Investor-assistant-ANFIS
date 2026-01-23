import React, { useState } from 'react';

const CollapsibleSection = ({ title, children, defaultOpen = true }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div style={styles.card}>
      <div style={styles.header} onClick={() => setIsOpen(!isOpen)}>
        <span style={styles.title}>{title}</span>
        <span style={styles.icon}>{isOpen ? 'â–¼' : 'â–¶'}</span>
      </div>
      {isOpen && <div style={styles.content}>{children}</div>}
    </div>
  );
};

const styles = {
  card: { backgroundColor: '#1e222d', borderRadius: '8px', marginBottom: '20px', boxShadow: '0 4px 6px rgba(0,0,0,0.3)', border: '1px solid #363a45', overflow: 'hidden' },
  header: { padding: '15px 20px', background: '#2a2e39', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #363a45', userSelect: 'none' },
  title: { fontWeight: 'bold', fontSize: '1.1em', color: '#d1d4dc' },
  icon: { color: '#2962ff', fontSize: '0.8em' },
  content: { padding: '20px' }
};

export default CollapsibleSection;