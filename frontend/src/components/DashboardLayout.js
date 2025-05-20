import React from 'react';
import DashboardContent from './DashboardContent';

const DashboardLayout = ({ 
  serverStatus, 
  modelInfo, 
  loading, 
  error 
}) => {
  // Render loading state
  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-cube">
          <div className="cube-face"></div>
        </div>
        <div className="loading-text">Loading Model Analysis Dashboard</div>
      </div>
    );
  }

  // Render error state
  if (error) {
    return (
      <div className="error-message">
        <div className="error-title">Connection Error</div>
        <div className="error-details">{error}</div>
        <div className="card">
          <h2 className="card-title">Troubleshooting Steps:</h2>
          <ol style={{ paddingLeft: '1.5rem' }}>
            <li>Make sure the API server is running on port 8000</li>
            <li>Check if CORS is properly enabled on the server</li>
            <li>Try running one of the example scripts like examples/mnist_demo.py</li>
            <li>Check for any error messages in the server console</li>
          </ol>
        </div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo-cube">
              <div className="cube-face">B</div>
            </div>
            <div>
              <h1 className="title">Model Analysis</h1>
              <p className="subtitle">
                {modelInfo 
                  ? `Connected to ${modelInfo.name} (${modelInfo.framework})`
                  : 'No model connected'}
              </p>
            </div>
          </div>
          <div className="header-buttons">
            <div className="status-indicator">
              <div className={`status-dot ${serverStatus?.status === 'online' ? 'status-online' : 'status-offline'}`}></div>
              <span className="status-text">
                {serverStatus?.status === 'online' ? 'API Connected' : 'API Disconnected'}
              </span>
            </div>
            <button className="primary-button">Run Analysis</button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <DashboardContent 
          serverStatus={serverStatus}
          modelInfo={modelInfo}
        />
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-text">
            Model Analysis Dashboard v1.0.0
          </div>
          <div className="footer-text">
            {serverStatus?.started_at && `Server started: ${new Date(serverStatus.started_at).toLocaleString()}`}
          </div>
        </div>
      </footer>
    </div>
  );
};

export default DashboardLayout;