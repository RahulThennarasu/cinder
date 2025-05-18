// src/components/ModelInfoCard.js
import React from 'react';
import './ModelInfoCard.css';

const ModelInfoCard = ({ modelInfo, isLoading, error }) => {
  if (isLoading) {
    return (
      <div className="card model-info-card">
        <div className="card-header">
          <h3 className="card-title">Model Information</h3>
        </div>
        <div className="card-body loading-container">
          <div className="loading-spinner"></div>
          <p className="loading-text">Loading model information...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card model-info-card error">
        <div className="card-header">
          <h3 className="card-title">Model Information</h3>
        </div>
        <div className="card-body">
          <div className="error-message">
            <p>Error loading model information:</p>
            <p className="error-details">{error.message}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!modelInfo) {
    return (
      <div className="card model-info-card empty">
        <div className="card-header">
          <h3 className="card-title">Model Information</h3>
        </div>
        <div className="card-body">
          <p>No model information available. Connect a model to view details.</p>
        </div>
      </div>
    );
  }

  // Calculate accuracy badge class
  const getAccuracyBadgeClass = (accuracy) => {
    if (accuracy >= 0.9) return 'badge-success';
    if (accuracy >= 0.7) return 'badge-warning';
    return 'badge-danger';
  };

  return (
    <div className="card model-info-card">
      <div className="card-header">
        <h3 className="card-title">Model Information</h3>
      </div>
      <div className="card-body">
        <div className="model-name">
          <h2>{modelInfo.name}</h2>
          <span className={`framework-badge ${modelInfo.framework.toLowerCase()}`}>
            {modelInfo.framework}
          </span>
        </div>
        
        <div className="info-grid">
          <div className="info-item">
            <div className="info-label">Dataset Size</div>
            <div className="info-value">{modelInfo.dataset_size.toLocaleString()}</div>
          </div>
          
          <div className="info-item accuracy">
            <div className="info-label">Accuracy</div>
            <div className="info-value">
              <span className={`badge ${getAccuracyBadgeClass(modelInfo.accuracy)}`}>
                {(modelInfo.accuracy * 100).toFixed(2)}%
              </span>
            </div>
          </div>
        </div>
        
        <div className="actions">
          <button className="btn btn-primary btn-sm">View Details</button>
          <button className="btn btn-secondary btn-sm">Export Report</button>
        </div>
      </div>
    </div>
  );
};

export default ModelInfoCard;