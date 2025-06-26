import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import DashboardLayout from './components/DashboardLayout';
import './index.css';
import './styles.css';

const App = () => {
  // State management
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [serverStatus, setServerStatus] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);

  // Fetch basic data from API
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const API_BASE_URL = 'http://localhost:8000/api';
        
        // Server status
        const statusResponse = await fetch(`${API_BASE_URL}/status`);
        const statusData = await statusResponse.json();
        setServerStatus(statusData);

        try {
          // Model info
          const modelResponse = await fetch(`${API_BASE_URL}/model`);
          const modelData = await modelResponse.json();
          setModelInfo(modelData);
        } catch (e) {
          console.error("Error fetching model data:", e);
          // Continue even if model data fails - show server status at minimum
        }
      } catch (e) {
        console.error("Fatal error fetching data:", e);
        setError("Could not connect to the API server. Make sure it's running on http://localhost:8000");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return (
    <DashboardLayout
      loading={loading}
      error={error}
      serverStatus={serverStatus}
      modelInfo={modelInfo}
    />
  );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);