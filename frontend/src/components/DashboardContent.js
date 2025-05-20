import React, { useState, useEffect } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell
} from 'recharts';

// Constants
const COLORS = ['#e74c32', '#ff9066', '#ffba66', '#ffd166', '#8884d8', '#82ca9d'];

const DashboardContent = ({ serverStatus, modelInfo }) => {
  // State management
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
      const API_BASE_URL = 'http://localhost:8000/api';
      
      try {
        // Only fetch if we have a server connection
        if (serverStatus && serverStatus.status === 'online') {
          // Confusion matrix
          try {
            const confusionResponse = await fetch(`${API_BASE_URL}/confusion-matrix`);
            const confusionData = await confusionResponse.json();
            setConfusionMatrix(confusionData);
          } catch (e) {
            console.error("Error fetching confusion matrix:", e);
          }

          // Feature importance
          try {
            const importanceResponse = await fetch(`${API_BASE_URL}/feature-importance`);
            const importanceData = await importanceResponse.json();
            setFeatureImportance(importanceData);
          } catch (e) {
            console.error("Error fetching feature importance:", e);
          }

          // Training history
          try {
            const historyResponse = await fetch(`${API_BASE_URL}/training-history`);
            const historyData = await historyResponse.json();
            setTrainingHistory(historyData);
          } catch (e) {
            console.error("Error fetching training history:", e);
          }

          // Prediction distribution
          try {
            const distributionResponse = await fetch(`${API_BASE_URL}/prediction-distribution`);
            const distributionData = await distributionResponse.json();
            setPredictionDistribution(distributionData);
          } catch (e) {
            console.error("Error fetching prediction distribution:", e);
          }

          // Error analysis
          try {
            const errorResponse = await fetch(`${API_BASE_URL}/errors`);
            const errorData = await errorResponse.json();
            setErrorAnalysis(errorData);
          } catch (e) {
            console.error("Error fetching error analysis:", e);
          }

          // Confidence analysis
          try {
            const confidenceResponse = await fetch(`${API_BASE_URL}/confidence-analysis`);
            const confidenceData = await confidenceResponse.json();
            setConfidenceAnalysis(confidenceData);
          } catch (e) {
            console.error("Error fetching confidence analysis:", e);
          }
        }
      } catch (e) {
        console.error("Error fetching data:", e);
      }
    };

    fetchData();
  }, [serverStatus]);

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

  // Helper function for confusion matrix visualization
  const renderConfusionMatrix = () => {
    if (!confusionMatrix) return <div className="empty-state">Confusion matrix data unavailable</div>;

    return (
      <div className="card">
        <h3 className="card-title">Confusion Matrix</h3>
        <div className="table-container">
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
                        ? `rgba(255, 123, 0, ${opacity})`
                        : `rgba(231, 76, 50, ${opacity*0.7})`
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

  // Render tabs
  const renderTabs = () => {
    const tabs = [
      { id: 'overview', label: 'Overview' },
      { id: 'performance', label: 'Performance' },
      { id: 'errors', label: 'Error Analysis' },
      { id: 'features', label: 'Features' },
      { id: 'training', label: 'Training' },
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
                <div className="empty-state">No model connected</div>
              )}
            </div>

            {/* Server Status */}
            <div className="card">
              <h3 className="card-title">Server Status</h3>
              {serverStatus && (
                <div>
                  <div className="info-row">
                    <span className="info-label">Status:</span>
                    <span className="info-value status-value">{serverStatus.status}</span>
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
                        className="progress-fill" 
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
                          className="progress-fill" 
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
                          className="progress-fill" 
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
                          className="progress-fill" 
                          style={{ width: `${modelInfo.f1 * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="empty-state">No model connected</div>
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
                      <Bar dataKey="count" fill="#e74c32" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="empty-state">Prediction distribution unavailable</div>
              )}
            </div>
          </div>
        );

      case 'performance':
        return (
          <div className="grid">
            {/* Performance Metrics Table */}
            <div className="card">
              <h3 className="card-title">Performance Metrics</h3>
              {modelInfo ? (
                <div className="table-container">
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
                          <div className="table-label">Accuracy</div>
                          <div className="table-sublabel">Overall model accuracy</div>
                        </td>
                        <td>
                          <div className="table-value">{(modelInfo.accuracy * 100).toFixed(2)}%</div>
                        </td>
                        <td style={{ width: '33%' }}>
                          <div className="progress-bar">
                            <div 
                              className="progress-fill" 
                              style={{ width: `${modelInfo.accuracy * 100}%` }}
                            ></div>
                          </div>
                        </td>
                      </tr>
                      {modelInfo.precision !== undefined && (
                        <tr>
                          <td>
                            <div className="table-label">Precision</div>
                            <div className="table-sublabel">Positive prediction accuracy</div>
                          </td>
                          <td>
                            <div className="table-value">{(modelInfo.precision * 100).toFixed(2)}%</div>
                          </td>
                          <td>
                            <div className="progress-bar">
                              <div 
                                className="progress-fill" 
                                style={{ width: `${modelInfo.precision * 100}%` }}
                              ></div>
                            </div>
                          </td>
                        </tr>
                      )}
                      {modelInfo.recall !== undefined && (
                        <tr>
                          <td>
                            <div className="table-label">Recall</div>
                            <div className="table-sublabel">Positive identification rate</div>
                          </td>
                          <td>
                            <div className="table-value">{(modelInfo.recall * 100).toFixed(2)}%</div>
                          </td>
                          <td>
                            <div className="progress-bar">
                              <div 
                                className="progress-fill" 
                                style={{ width: `${modelInfo.recall * 100}%` }}
                              ></div>
                            </div>
                          </td>
                        </tr>
                      )}
                      {modelInfo.f1 !== undefined && (
                        <tr>
                          <td>
                            <div className="table-label">F1 Score</div>
                            <div className="table-sublabel">Balance of precision and recall</div>
                          </td>
                          <td>
                            <div className="table-value">{(modelInfo.f1 * 100).toFixed(2)}%</div>
                          </td>
                          <td>
                            <div className="progress-bar">
                              <div 
                                className="progress-fill" 
                                style={{ width: `${modelInfo.f1 * 100}%` }}
                              ></div>
                            </div>
                          </td>
                        </tr>
                      )}
                      {modelInfo.roc_auc !== undefined && (
                        <tr>
                          <td>
                            <div className="table-label">ROC AUC</div>
                            <div className="table-sublabel">Area under ROC curve</div>
                          </td>
                          <td>
                            <div className="table-value">{(modelInfo.roc_auc * 100).toFixed(2)}%</div>
                          </td>
                          <td>
                            <div className="progress-bar">
                              <div 
                                className="progress-fill" 
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
                <div className="empty-state">No model metrics available</div>
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
                  <div className="stats-grid">
                    <div className="stat-box">
                      <div className="stat-label">Error Count</div>
                      <div className="stat-value">{errorAnalysis.error_count}</div>
                    </div>
                    <div className="stat-box">
                      <div className="stat-label">Error Rate</div>
                      <div className="stat-value">{(errorAnalysis.error_rate * 100).toFixed(2)}%</div>
                    </div>
                  </div>
                  
                  <div className="chart-section">
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
                            fill="#e74c32"
                            dataKey="value"
                            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                          >
                            <Cell fill="#7dd3fc" />
                            <Cell fill="#e74c32" />
                          </Pie>
                          <Tooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="empty-state">Error analysis unavailable</div>
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
                            <div className="table-label">
                              {`Class ${type.true_class} → ${type.predicted_class}`}
                            </div>
                          </td>
                          <td>
                            <div className="table-value">{type.count}</div>
                          </td>
                          <td>
                            <div className="table-value">
                              {((type.count / errorAnalysis.error_count) * 100).toFixed(2)}%
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="empty-state">Error type analysis unavailable</div>
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
                  <div className="method-info">
                    <span className="method-label">Method: </span>
                    <span className="method-value">{featureImportance.importance_method}</span>
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
                        <Bar dataKey="importance" fill="#e74c32" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              ) : (
                <div className="empty-state">Feature importance unavailable</div>
              )}
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
                      <Line yAxisId="left" type="monotone" dataKey="accuracy" stroke="#e74c32" name="Accuracy" />
                      <Line yAxisId="right" type="monotone" dataKey="loss" stroke="#7dd3fc" name="Loss" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="empty-state">No training history available</div>
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
                      <Line type="monotone" dataKey="learning_rate" stroke="#e74c32" name="Learning Rate" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="empty-state">No learning rate data available</div>
              )}
            </div>
          </div>
        );

      default:
        return <div>Select a tab to view data</div>;
    }
  };

  return (
    <div>
      {renderTabs()}
      {renderTabContent()}
    </div>
  );
};

export default DashboardContent;