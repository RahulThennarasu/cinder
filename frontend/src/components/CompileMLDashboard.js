import React, { useState, useEffect } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line, AreaChart, Area, PieChart, Pie, Cell,
  ScatterChart, Scatter, ZAxis
} from 'recharts';

// Constants
const API_BASE_URL = 'http://localhost:8000/api';
const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#8dd1e1'];

// Main Dashboard Component
const CompileMLDashboard = () => {
  // State management
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [serverStatus, setServerStatus] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [confusionMatrix, setConfusionMatrix] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState(null);
  const [predictionDistribution, setPredictionDistribution] = useState(null);
  const [errorAnalysis, setErrorAnalysis] = useState(null);
  const [confidenceAnalysis, setConfidenceAnalysis] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  // Fetch data from API
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Server status
        const statusResponse = await fetch(`${API_BASE_URL}/status`);
        const statusData = await statusResponse.json();
        setServerStatus(statusData);

        try {
          // Model info
          const modelResponse = await fetch(`${API_BASE_URL}/model`);
          const modelData = await modelResponse.json();
          setModelInfo(modelData);

          // Confusion matrix
          const confusionResponse = await fetch(`${API_BASE_URL}/confusion-matrix`);
          const confusionData = await confusionResponse.json();
          setConfusionMatrix(confusionData);

          // Feature importance
          const importanceResponse = await fetch(`${API_BASE_URL}/feature-importance`);
          const importanceData = await importanceResponse.json();
          setFeatureImportance(importanceData);

          // Training history
          const historyResponse = await fetch(`${API_BASE_URL}/training-history`);
          const historyData = await historyResponse.json();
          setTrainingHistory(historyData);

          // Prediction distribution
          const distributionResponse = await fetch(`${API_BASE_URL}/prediction-distribution`);
          const distributionData = await distributionResponse.json();
          setPredictionDistribution(distributionData);

          // Error analysis
          const errorResponse = await fetch(`${API_BASE_URL}/errors`);
          const errorData = await errorResponse.json();
          setErrorAnalysis(errorData);

          // Confidence analysis
          const confidenceResponse = await fetch(`${API_BASE_URL}/confidence-analysis`);
          const confidenceData = await confidenceResponse.json();
          setConfidenceAnalysis(confidenceData);
        } catch (e) {
          console.error("Error fetching model data:", e);
          // Continue even if model data fails - show server status at minimum
        }
      } catch (e) {
        console.error("Fatal error fetching data:", e);
        setError("Could not connect to the CompileML server. Make sure it's running on http://localhost:8000");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Render loading state
  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-text">Loading CompileML Dashboard</div>
        <div className="spinner"></div>
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
            <li>Make sure the CompileML server is running on port 8000</li>
            <li>Check if CORS is properly enabled on the server</li>
            <li>Try running one of the example scripts like examples/mnist_demo.py</li>
            <li>Check for any error messages in the server console</li>
          </ol>
        </div>
      </div>
    );
  }

  // Helper function for confusion matrix visualization
  const renderConfusionMatrix = () => {
    if (!confusionMatrix) return <div style={{ color: '#6b7280' }}>Confusion matrix data unavailable</div>;

    return (
      <div className="card">
        <h3 className="card-title">Confusion Matrix</h3>
        <div style={{ overflowX: 'auto' }}>
          <table className="table confusion-matrix">
            <thead>
              <tr>
                <th></th>
                {confusionMatrix.labels.map((label, idx) => (
                  <th key={idx}>
                    Predicted: {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {confusionMatrix.matrix.map((row, rowIdx) => (
                <tr key={rowIdx}>
                  <th>
                    Actual: {confusionMatrix.labels[rowIdx]}
                  </th>
                  {row.map((cell, cellIdx) => {
                    const isCorrect = rowIdx === cellIdx;
                    const opacity = cell / Math.max(...confusionMatrix.matrix.flat());
                    const style = {
                      backgroundColor: isCorrect 
                        ? `rgba(16, 185, 129, ${opacity})`
                        : `rgba(239, 68, 68, ${opacity})`
                    };
                    
                    return (
                      <td 
                        key={cellIdx}
                        className={isCorrect ? 'correct-cell' : 'incorrect-cell'}
                        style={style}
                      >
                        {cell}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  // Prepare feature importance data for chart
  const prepareFeatureImportanceData = () => {
    if (!featureImportance || !featureImportance.feature_names) return [];
    
    return featureImportance.feature_names.map((name, idx) => ({
      name: name,
      importance: featureImportance.importance_values[idx]
    })).sort((a, b) => b.importance - a.importance).slice(0, 10);
  };

  // Prepare confidence distribution data
  const prepareConfidenceData = () => {
    if (!confidenceAnalysis || !confidenceAnalysis.confidence_distribution) return [];
    
    const { bin_edges, overall, correct, incorrect } = confidenceAnalysis.confidence_distribution;
    
    return bin_edges.slice(0, -1).map((edge, idx) => ({
      range: `${(edge * 100).toFixed(0)}-${(bin_edges[idx+1] * 100).toFixed(0)}%`,
      overall: overall[idx],
      correct: correct[idx],
      incorrect: incorrect[idx]
    }));
  };

  // Render tabs
  const renderTabs = () => {
    const tabs = [
      { id: 'overview', label: 'Overview' },
      { id: 'metrics', label: 'Performance Metrics' },
      { id: 'errors', label: 'Error Analysis' },
      { id: 'features', label: 'Feature Importance' },
      { id: 'confidence', label: 'Confidence Analysis' },
      { id: 'predictions', label: 'Predictions' },
      { id: 'training', label: 'Training History' },
    ];

    return (
      <div className="tabs">
        <nav className="tab-list">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>
    );
  };

  // Render based on active tab
  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <div className="grid grid-cols-2">
            {/* Model Info */}
            <div className="card">
              <h3 className="card-title">Model Information</h3>
              {modelInfo ? (
                <div>
                  <div className="info-row">
                    <span className="info-label">Name:</span>
                    <span className="info-value">{modelInfo.name}</span>
                  </div>
                  <div className="info-row">
                    <span className="info-label">Framework:</span>
                    <span className="info-value">{modelInfo.framework}</span>
                  </div>
                  <div className="info-row">
                    <span className="info-label">Dataset Size:</span>
                    <span className="info-value">{modelInfo.dataset_size} samples</span>
                  </div>
                </div>
              ) : (
                <div style={{ color: '#6b7280' }}>No model connected</div>
              )}
            </div>

            {/* Server Status */}
            <div className="card">
              <h3 className="card-title">Server Status</h3>
              {serverStatus && (
                <div>
                  <div className="info-row">
                    <span className="info-label">Status:</span>
                    <span className="info-value" style={{ color: '#10b981' }}>{serverStatus.status}</span>
                  </div>
                  <div className="info-row">
                    <span className="info-label">Uptime:</span>
                    <span className="info-value">{serverStatus.uptime}</span>
                  </div>
                  <div className="info-row">
                    <span className="info-label">Version:</span>
                    <span className="info-value">{serverStatus.version}</span>
                  </div>
                  <div className="info-row">
                    <span className="info-label">Memory Usage:</span>
                    <span className="info-value">{serverStatus.memory_usage?.toFixed(2)} MB</span>
                  </div>
                </div>
              )}
            </div>

            {/* Model Performance */}
            <div className="card">
              <h3 className="card-title">Model Performance</h3>
              {modelInfo ? (
                <div>
                  <div className="metric-container">
                    <div className="metric-header">
                      <span className="metric-label">Accuracy</span>
                      <span className="metric-value">{(modelInfo.accuracy * 100).toFixed(2)}%</span>
                    </div>
                    <div className="progress-bar">
                      <div 
                        className="progress-fill progress-fill-primary" 
                        style={{ width: `${modelInfo.accuracy * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  {modelInfo.precision && (
                    <div className="metric-container">
                      <div className="metric-header">
                        <span className="metric-label">Precision</span>
                        <span className="metric-value">{(modelInfo.precision * 100).toFixed(2)}%</span>
                      </div>
                      <div className="progress-bar">
                        <div 
                          className="progress-fill progress-fill-blue" 
                          style={{ width: `${modelInfo.precision * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  )}

                  {modelInfo.recall && (
                    <div className="metric-container">
                      <div className="metric-header">
                        <span className="metric-label">Recall</span>
                        <span className="metric-value">{(modelInfo.recall * 100).toFixed(2)}%</span>
                      </div>
                      <div className="progress-bar">
                        <div 
                          className="progress-fill progress-fill-green" 
                          style={{ width: `${modelInfo.recall * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  )}

                  {modelInfo.f1 && (
                    <div className="metric-container">
                      <div className="metric-header">
                        <span className="metric-label">F1 Score</span>
                        <span className="metric-value">{(modelInfo.f1 * 100).toFixed(2)}%</span>
                      </div>
                      <div className="progress-bar">
                        <div 
                          className="progress-fill progress-fill-purple" 
                          style={{ width: `${modelInfo.f1 * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ color: '#6b7280' }}>No model connected</div>
              )}
            </div>

            {/* Prediction Distribution */}
            <div className="card">
              <h3 className="card-title">Prediction Distribution</h3>
              {predictionDistribution ? (
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={predictionDistribution}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="class_name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="count" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div style={{ color: '#6b7280' }}>Prediction distribution unavailable</div>
              )}
            </div>
          </div>
        );

      case 'metrics':
        return (
          <div className="grid">
            {/* Performance Metrics Table */}
            <div className="card">
              <h3 className="card-title">Performance Metrics</h3>
              {modelInfo ? (
                <div style={{ overflowX: 'auto' }}>
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Visualization</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td>
                          <div style={{ fontSize: '0.875rem', fontWeight: '500', color: '#111827' }}>Accuracy</div>
                          <div style={{ fontSize: '0.875rem', color: '#6b7280' }}>Overall model accuracy</div>
                        </td>
                        <td>
                          <div style={{ fontSize: '0.875rem', color: '#111827', fontWeight: '500' }}>{(modelInfo.accuracy * 100).toFixed(2)}%</div>
                        </td>
                        <td style={{ width: '33%' }}>
                          <div className="progress-bar">
                            <div 
                              className="progress-fill progress-fill-primary" 
                              style={{ width: `${modelInfo.accuracy * 100}%` }}
                            ></div>
                          </div>
                        </td>
                      </tr>
                      {modelInfo.precision !== undefined && (
                        <tr>
                          <td>
                            <div style={{ fontSize: '0.875rem', fontWeight: '500', color: '#111827' }}>Precision</div>
                            <div style={{ fontSize: '0.875rem', color: '#6b7280' }}>Positive prediction accuracy</div>
                          </td>
                          <td>
                            <div style={{ fontSize: '0.875rem', color: '#111827', fontWeight: '500' }}>{(modelInfo.precision * 100).toFixed(2)}%</div>
                          </td>
                          <td>
                            <div className="progress-bar">
                              <div 
                                className="progress-fill progress-fill-blue" 
                                style={{ width: `${modelInfo.precision * 100}%` }}
                              ></div>
                            </div>
                          </td>
                        </tr>
                      )}
                      {modelInfo.recall !== undefined && (
                        <tr>
                          <td>
                            <div style={{ fontSize: '0.875rem', fontWeight: '500', color: '#111827' }}>Recall</div>
                            <div style={{ fontSize: '0.875rem', color: '#6b7280' }}>Positive identification rate</div>
                          </td>
                          <td>
                            <div style={{ fontSize: '0.875rem', color: '#111827', fontWeight: '500' }}>{(modelInfo.recall * 100).toFixed(2)}%</div>
                          </td>
                          <td>
                            <div className="progress-bar">
                              <div 
                                className="progress-fill progress-fill-green" 
                                style={{ width: `${modelInfo.recall * 100}%` }}
                              ></div>
                            </div>
                          </td>
                        </tr>
                      )}
                      {modelInfo.f1 !== undefined && (
                        <tr>
                          <td>
                            <div style={{ fontSize: '0.875rem', fontWeight: '500', color: '#111827' }}>F1 Score</div>
                            <div style={{ fontSize: '0.875rem', color: '#6b7280' }}>Balance of precision and recall</div>
                          </td>
                          <td>
                            <div style={{ fontSize: '0.875rem', color: '#111827', fontWeight: '500' }}>{(modelInfo.f1 * 100).toFixed(2)}%</div>
                          </td>
                          <td>
                            <div className="progress-bar">
                              <div 
                                className="progress-fill progress-fill-purple" 
                                style={{ width: `${modelInfo.f1 * 100}%` }}
                              ></div>
                            </div>
                          </td>
                        </tr>
                      )}
                      {modelInfo.roc_auc !== undefined && (
                        <tr>
                          <td>
                            <div style={{ fontSize: '0.875rem', fontWeight: '500', color: '#111827' }}>ROC AUC</div>
                            <div style={{ fontSize: '0.875rem', color: '#6b7280' }}>Area under ROC curve</div>
                          </td>
                          <td>
                            <div style={{ fontSize: '0.875rem', color: '#111827', fontWeight: '500' }}>{(modelInfo.roc_auc * 100).toFixed(2)}%</div>
                          </td>
                          <td>
                            <div className="progress-bar">
                              <div 
                                className="progress-fill progress-fill-yellow" 
                                style={{ width: `${modelInfo.roc_auc * 100}%` }}
                              ></div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div style={{ color: '#6b7280' }}>No model metrics available</div>
              )}
            </div>

            {/* Confusion Matrix */}
            {renderConfusionMatrix()}
          </div>
        );

      case 'errors':
        return (
          <div className="grid grid-cols-2">
            {/* Error Analysis */}
            <div className="card">
              <h3 className="card-title">Error Analysis</h3>
              {errorAnalysis ? (
                <div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <div className="stat-box">
                      <div className="stat-label">Error Count</div>
                      <div className="stat-value">{errorAnalysis.error_count}</div>
                    </div>
                    <div className="stat-box">
                      <div className="stat-label">Error Rate</div>
                      <div className="stat-value">{(errorAnalysis.error_rate * 100).toFixed(2)}%</div>
                    </div>
                  </div>
                  
                  <div style={{ marginTop: '1rem' }}>
                    <div className="metric-header">
                      <span className="metric-label">Correct vs. Errors</span>
                    </div>
                    <div className="chart-container">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={[
                              { name: 'Correct', value: errorAnalysis.correct_count },
                              { name: 'Errors', value: errorAnalysis.error_count }
                            ]}
                            cx="50%"
                            cy="50%"
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                          >
                            <Cell fill="#4ade80" />
                            <Cell fill="#f87171" />
                          </Pie>
                          <Tooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ color: '#6b7280' }}>Error analysis unavailable</div>
              )}
            </div>

            {/* Error Types */}
            <div className="card">
              <h3 className="card-title">Error Types</h3>
              {errorAnalysis && errorAnalysis.error_types ? (
                <div className="scrollable">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Error Type</th>
                        <th>Count</th>
                        <th>Percentage</th>
                      </tr>
                    </thead>
                    <tbody>
                      {errorAnalysis.error_types.map((type, idx) => (
                        <tr key={idx}>
                          <td>
                            <div style={{ fontSize: '0.875rem', fontWeight: '500', color: '#111827' }}>
                              {`Class ${type.true_class} → ${type.predicted_class}`}
                            </div>
                          </td>
                          <td>
                            <div style={{ fontSize: '0.875rem', color: '#111827' }}>{type.count}</div>
                          </td>
                          <td>
                            <div style={{ fontSize: '0.875rem', color: '#111827' }}>
                              {((type.count / errorAnalysis.error_count) * 100).toFixed(2)}%
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div style={{ color: '#6b7280' }}>Error type analysis unavailable</div>
              )}
            </div>
          </div>
        );

      case 'features':
        return (
          <div className="grid">
            {/* Feature Importance */}
            <div className="card">
              <h3 className="card-title">Feature Importance</h3>
              {featureImportance ? (
                <div>
                  <div style={{ fontSize: '0.875rem', marginBottom: '1rem' }}>
                    <span style={{ fontWeight: '500' }}>Method: </span>
                    <span style={{ color: '#4b5563' }}>{featureImportance.importance_method}</span>
                  </div>
                  <div className="chart-container-tall">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={prepareFeatureImportanceData()}
                        layout="vertical"
                        margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" domain={[0, 'dataMax']} />
                        <YAxis dataKey="name" type="category" width={100} />
                        <Tooltip formatter={(value) => [(value * 100).toFixed(2) + '%', 'Importance']} />
                        <Bar dataKey="importance" fill="#8884d8" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              ) : (
                <div style={{ color: '#6b7280' }}>Feature importance unavailable</div>
              )}
            </div>
          </div>
        );

      case 'confidence':
        return (
          <div className="grid grid-cols-2">
            {/* Confidence Analysis */}
            <div className="card">
              <h3 className="card-title">Confidence Metrics</h3>
              {confidenceAnalysis ? (
                <div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <div className="stat-box">
                      <div className="stat-label">Average Confidence</div>
                      <div className="stat-value">{(confidenceAnalysis.avg_confidence * 100).toFixed(2)}%</div>
                    </div>
                    <div className="stat-box">
                      <div className="stat-label">Calibration Error</div>
                      <div className="stat-value">{(confidenceAnalysis.calibration_error * 100).toFixed(2)}%</div>
                    </div>
                  </div>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '1rem' }}>
                    <div className="stat-box">
                      <div className="stat-label">Avg. Correct Confidence</div>
                      <div className="stat-value">{(confidenceAnalysis.avg_correct_confidence * 100).toFixed(2)}%</div>
                    </div>
                    <div className="stat-box">
                      <div className="stat-label">Avg. Incorrect Confidence</div>
                      <div className="stat-value">{(confidenceAnalysis.avg_incorrect_confidence * 100).toFixed(2)}%</div>
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ color: '#6b7280' }}>Confidence analysis unavailable</div>
              )}
            </div>

            {/* Confidence Distribution */}
            <div className="card">
              <h3 className="card-title">Confidence Distribution</h3>
              {confidenceAnalysis && confidenceAnalysis.confidence_distribution ? (
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={prepareConfidenceData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="range" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="correct" name="Correct Predictions" fill="#4ade80" />
                      <Bar dataKey="incorrect" name="Incorrect Predictions" fill="#f87171" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div style={{ color: '#6b7280' }}>Confidence distribution unavailable</div>
              )}
            </div>

            {/* Overconfident Examples */}
            <div className="card">
              <h3 className="card-title">Overconfident Predictions</h3>
              {confidenceAnalysis && confidenceAnalysis.overconfident_examples ? (
                <div>
                  <div style={{ marginBottom: '1rem' }}>
                    <span className="info-label">Threshold: </span>
                    <span className="info-value">{confidenceAnalysis.overconfident_examples.threshold * 100}%</span>
                  </div>
                  <div style={{ marginBottom: '1rem' }}>
                    <span className="info-label">Count: </span>
                    <span className="info-value">{confidenceAnalysis.overconfident_examples.count} examples</span>
                  </div>
                  <div className="alert alert-danger">
                    <p>
                      These are examples where the model was highly confident but wrong, indicating potential blind spots.
                    </p>
                  </div>
                </div>
              ) : (
                <div style={{ color: '#6b7280' }}>No overconfident examples data</div>
              )}
            </div>

            {/* Underconfident Examples */}
            <div className="card">
              <h3 className="card-title">Underconfident Predictions</h3>
              {confidenceAnalysis && confidenceAnalysis.underconfident_examples ? (
                <div>
                  <div style={{ marginBottom: '1rem' }}>
                    <span className="info-label">Threshold: </span>
                    <span className="info-value">{confidenceAnalysis.underconfident_examples.threshold * 100}%</span>
                  </div>
                  <div style={{ marginBottom: '1rem' }}>
                    <span className="info-label">Count: </span>
                    <span className="info-value">{confidenceAnalysis.underconfident_examples.count} examples</span>
                  </div>
                  <div className="alert alert-warning">
                    <p>
                      These are examples where the model was correct but had low confidence, indicating room for improved calibration.
                    </p>
                  </div>
                </div>
              ) : (
                <div style={{ color: '#6b7280' }}>No underconfident examples data</div>
              )}
            </div>
          </div>
        );

      case 'predictions':
        return (
          <div className="grid">
            {/* Sample Predictions - Mocked for now, would connect to API in real implementation */}
            <div className="card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h3 className="card-title" style={{ margin: 0 }}>Sample Predictions</h3>
                <button className="button button-primary">
                  Fetch Samples
                </button>
              </div>
              
              <div className="form-controls">
                <div className="form-group">
                  <input type="checkbox" id="errors-only" className="form-checkbox" />
                  <label htmlFor="errors-only" className="form-label">Show errors only</label>
                </div>
                
                <div className="form-group">
                  <label htmlFor="limit" className="form-label">Limit:</label>
                  <select id="limit" className="form-select">
                    <option>10</option>
                    <option>20</option>
                    <option>50</option>
                    <option>100</option>
                  </select>
                </div>
              </div>
              
              <div style={{ overflowX: 'auto' }}>
                <table className="table">
                  <thead>
                    <tr>
                      <th>Sample ID</th>
                      <th>True Label</th>
                      <th>Predicted</th>
                      <th>Confidence</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {/* Demo data - would be replaced with real API data */}
                    {[...Array(10)].map((_, idx) => {
                      const isError = idx % 3 === 0;
                      return (
                        <tr key={idx} style={{ backgroundColor: isError ? '#fee2e2' : 'white' }}>
                          <td>
                            <div style={{ fontSize: '0.875rem', fontWeight: '500', color: '#111827' }}>#{idx + 1}</div>
                          </td>
                          <td>
                            <div style={{ fontSize: '0.875rem', color: '#111827' }}>{idx % 2}</div>
                          </td>
                          <td>
                            <div style={{ fontSize: '0.875rem', color: '#111827' }}>{isError ? (idx % 2 === 0 ? 1 : 0) : idx % 2}</div>
                          </td>
                          <td>
                            <div style={{ fontSize: '0.875rem', color: '#111827' }}>
                              {isError ? `${(0.7 + Math.random() * 0.2).toFixed(2)}` : `${(0.8 + Math.random() * 0.15).toFixed(2)}`}
                            </div>
                          </td>
                          <td>
                            <span className={`badge ${isError ? 'badge-error' : 'badge-success'}`}>
                              {isError ? 'Error' : 'Correct'}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              
              <div className="pagination">
                <div className="pagination-info">
                  Showing <span style={{ fontWeight: '500' }}>1</span> to <span style={{ fontWeight: '500' }}>10</span> of <span style={{ fontWeight: '500' }}>100</span> results
                </div>
                <div className="pagination-controls">
                  <button className="pagination-button" disabled>Previous</button>
                  <button className="pagination-button pagination-button-active">Next</button>
                </div>
              </div>
            </div>
          </div>
        );

      case 'training':
        return (
          <div className="grid">
            {/* Training History */}
            <div className="card">
              <h3 className="card-title">Training History</h3>
              {trainingHistory && trainingHistory.length > 0 ? (
                <div className="chart-container-tall">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trainingHistory}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="iteration" />
                      <YAxis yAxisId="left" label={{ value: 'Accuracy', angle: -90, position: 'insideLeft' }} />
                      <YAxis yAxisId="right" orientation="right" label={{ value: 'Loss', angle: 90, position: 'insideRight' }} />
                      <Tooltip formatter={(value, name) => [value.toFixed(4), name]} />
                      <Legend />
                      <Line yAxisId="left" type="monotone" dataKey="accuracy" stroke="#8884d8" name="Accuracy" />
                      <Line yAxisId="right" type="monotone" dataKey="loss" stroke="#82ca9d" name="Loss" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div style={{ color: '#6b7280' }}>No training history available</div>
              )}
            </div>
            
            {/* Learning Rate */}
            <div className="card">
              <h3 className="card-title">Learning Rate</h3>
              {trainingHistory && trainingHistory.length > 0 && trainingHistory[0].learning_rate ? (
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trainingHistory}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="iteration" />
                      <YAxis />
                      <Tooltip formatter={(value) => [value.toExponential(4), 'Learning Rate']} />
                      <Line type="monotone" dataKey="learning_rate" stroke="#ff7300" name="Learning Rate" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div style={{ color: '#6b7280' }}>No learning rate data available</div>
              )}
            </div>
          </div>
        );

      default:
        return <div>Select a tab to view data</div>;
    }
  };

  return (
    <div className="dashboard">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div>
            <h1 className="title">CompileML Dashboard</h1>
            <p className="subtitle">
              {modelInfo 
                ? `Connected to ${modelInfo.name} (${modelInfo.framework})`
                : 'No model connected'}
            </p>
          </div>
          <div className="status-indicator">
            <div className={`status-dot ${serverStatus?.status === 'online' ? 'status-online' : 'status-offline'}`}></div>
            <span className="status-text">
              {serverStatus?.status === 'online' ? 'Server Online' : 'Server Offline'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {renderTabs()}
        {renderTabContent()}
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <div className="footer-text">
            CompileML Dashboard v1.0.0
          </div>
          <div className="footer-text">
            {serverStatus?.started_at && `Server started: ${new Date(serverStatus.started_at).toLocaleString()}`}
          </div>
        </div>
      </footer>
    </div>
  );
};

export default CompileMLDashboard;