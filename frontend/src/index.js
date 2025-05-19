import React from 'react';
import ReactDOM from 'react-dom/client';
import CompileMLDashboard from './components/CompileMLDashboard';
import './styles.css';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <CompileMLDashboard />
  </React.StrictMode>
);