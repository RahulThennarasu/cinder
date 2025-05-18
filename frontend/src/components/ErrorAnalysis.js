// src/components/ErrorAnalysis.js
import React, { useState } from 'react';
import './ErrorAnalysis.css';

const ErrorAnalysis = ({ errorData, isLoading, error }) => {
  const [activeTab, setActiveTab] = useState('overview');

  if (isLoading) {
    return (
      <div className="card error-analysis-card">
        <div className="card-header">
          <h3 className="card-title">Error Analysis</h3>
        </div>
        <div className="card-body loading-container">
          <div className="loading-spinner"></div>
          <p className="loading-text">Loading error analysis...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card error-analysis-card error">
        <div className="card-header">
          <h3 className="card-title">Error Analysis</h3>
        </div>
        <div className="card-body">
          <div className="error-message">
            <p>Error loading error analysis:</p>
            <p className="error-details">{error.message}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!errorData) {
    return (
      <div className="card error-analysis-card empty">
        <div className="card-header">
          <h3 className="card-title">Error Analysis</h3>
        </div>
        <div className="card-body">
          <p>No error analysis available.</p>
        </div>
      </div>
    );
  }

  // Calculate error rate
  const errorRate = errorData.error_count / (errorData.error_count + (errorData.correct_count || 0));
  
  return (
    <div className="card error-analysis-card">
      <div className="card-header">
        <h3 className="card-title">Error Analysis</h3>
      </div>
      <div className="card-body">
        <div className="error-analysis-tabs">
          <button 
            className={`tab-button ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </button>
          <button 
            className={`tab-button ${activeTab === 'samples' ? 'active' : ''}`}
            onClick={() => setActiveTab('samples')}
          >
            Error Samples
          </button>
        </div>
        
        <div className="tab-content">
          {activeTab === 'overview' && (
            <div className="overview-tab">
              <div className="error-summary">
                <div className="error-metric">
                  <div className="metric-value">{errorData.error_count}</div>
                  <div className="metric-label">Total Errors</div>
                </div>
                
                <div className="error-metric">
                  <div className="metric-value">
                    <span className="error-rate-badge">
                      {(errorRate * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="metric-label">Error Rate</div>
                </div>
              </div>
              
              <div className="error-visualization">
                <div className="error-bar">
                  <div 
                    className="error-bar-fill" 
                    style={{ width: `${(errorRate * 100)}%` }}
                  ></div>
                </div>
              </div>
              
              <div className="actions">
                <button className="btn btn-primary btn-sm">Analyze Deeper</button>
                <button className="btn btn-secondary btn-sm">Export Data</button>
              </div>
            </div>
          )}
          
          {activeTab === 'samples' && (
            <div className="samples-tab">
              {errorData.error_indices && errorData.error_indices.length > 0 ? (
                <>
                  <div className="samples-header">
                    <h4>Error Samples</h4>
                    <span className="samples-count">
                      Showing {Math.min(errorData.error_indices.length, 10)} of {errorData.error_count}
                    </span>
                  </div>
                  
                  <div className="samples-list">
                    {errorData.error_indices.slice(0, 10).map((index, i) => (
                      <div key={i} className="sample-item">
                        <div className="sample-index">#{index}</div>
                        <div className="sample-actions">
                          <button className="btn btn-sm">View</button>
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  {errorData.error_indices.length > 10 && (
                    <div className="view-all">
                      <button className="btn btn-secondary btn-sm">
                        View All {errorData.error_count} Errors
                      </button>
                    </div>
                  )}
                </>
              ) : (
                <p>No error samples available.</p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ErrorAnalysis;