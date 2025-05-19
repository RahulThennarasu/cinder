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
      <div className="flex flex-col items-center justify-center h-screen bg-gray-50">
        <div className="text-3xl font-bold text-indigo-600 mb-4">Loading CompileML Dashboard</div>
        <div className="w-16 h-16 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  // Render error state
  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-gray-50 p-4">
        <div className="text-3xl font-bold text-red-600 mb-4">Connection Error</div>
        <div className="text-lg text-gray-700 mb-6">{error}</div>
        <div className="bg-white p-6 rounded-lg shadow-md w-full max-w-2xl">
          <h2 className="text-xl font-semibold mb-4">Troubleshooting Steps:</h2>
          <ol className="list-decimal pl-6 space-y-2">
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
    if (!confusionMatrix) return <div className="text-gray-500">Confusion matrix data unavailable</div>;

    return (
      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Confusion Matrix</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead>
              <tr>
                <th className="p-2 border bg-gray-100"></th>
                {confusionMatrix.labels.map((label, idx) => (
                  <th key={idx} className="p-2 border bg-gray-100">
                    Predicted: {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {confusionMatrix.matrix.map((row, rowIdx) => (
                <tr key={rowIdx}>
                  <th className="p-2 border bg-gray-100">
                    Actual: {confusionMatrix.labels[rowIdx]}
                  </th>
                  {row.map((cell, cellIdx) => (
                    <td 
                      key={cellIdx} 
                      className={`p-3 border text-center ${rowIdx === cellIdx ? 'bg-green-100' : 'bg-white'}`}
                      style={{
                        backgroundColor: rowIdx === cellIdx 
                          ? `rgba(0, 200, 0, ${cell / Math.max(...row.map(r => Math.max(...r)))})`
                          : `rgba(255, 100, 100, ${cell / Math.max(...row.map(r => Math.max(...r)))})`
                      }}
                    >
                      {cell}
                    </td>
                  ))}
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
      <div className="mb-6">
        <div className="border-b border-gray-200">
          <nav className="flex -mb-px">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-6 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-b-2 border-indigo-500 text-indigo-600'
                    : 'text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </div>
    );
  };

  // Render based on active tab
  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Model Info */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Model Information</h3>
              {modelInfo ? (
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Name:</span>
                    <span className="font-medium">{modelInfo.name}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Framework:</span>
                    <span className="font-medium">{modelInfo.framework}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Dataset Size:</span>
                    <span className="font-medium">{modelInfo.dataset_size} samples</span>
                  </div>
                </div>
              ) : (
                <div className="text-gray-500">No model connected</div>
              )}
            </div>

            {/* Server Status */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Server Status</h3>
              {serverStatus && (
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Status:</span>
                    <span className="font-medium text-green-500">{serverStatus.status}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Uptime:</span>
                    <span className="font-medium">{serverStatus.uptime}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Version:</span>
                    <span className="font-medium">{serverStatus.version}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Memory Usage:</span>
                    <span className="font-medium">{serverStatus.memory_usage?.toFixed(2)} MB</span>
                  </div>
                </div>
              )}
            </div>

            {/* Model Performance */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Model Performance</h3>
              {modelInfo ? (
                <div className="space-y-4">
                  <div className="w-full">
                    <div className="flex justify-between mb-1">
                      <span className="text-gray-600">Accuracy</span>
                      <span className="font-medium">{(modelInfo.accuracy * 100).toFixed(2)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                      <div 
                        className="bg-indigo-600 h-2.5 rounded-full" 
                        style={{ width: `${modelInfo.accuracy * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  {modelInfo.precision && (
                    <div className="w-full">
                      <div className="flex justify-between mb-1">
                        <span className="text-gray-600">Precision</span>
                        <span className="font-medium">{(modelInfo.precision * 100).toFixed(2)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div 
                          className="bg-blue-500 h-2.5 rounded-full" 
                          style={{ width: `${modelInfo.precision * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  )}

                  {modelInfo.recall && (
                    <div className="w-full">
                      <div className="flex justify-between mb-1">
                        <span className="text-gray-600">Recall</span>
                        <span className="font-medium">{(modelInfo.recall * 100).toFixed(2)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div 
                          className="bg-green-500 h-2.5 rounded-full" 
                          style={{ width: `${modelInfo.recall * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  )}

                  {modelInfo.f1 && (
                    <div className="w-full">
                      <div className="flex justify-between mb-1">
                        <span className="text-gray-600">F1 Score</span>
                        <span className="font-medium">{(modelInfo.f1 * 100).toFixed(2)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div 
                          className="bg-purple-500 h-2.5 rounded-full" 
                          style={{ width: `${modelInfo.f1 * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-gray-500">No model connected</div>
              )}
            </div>

            {/* Prediction Distribution */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Prediction Distribution</h3>
              {predictionDistribution ? (
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={predictionDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="class_name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-gray-500">Prediction distribution unavailable</div>
              )}
            </div>
          </div>
        );

      case 'metrics':
        return (
          <div className="grid grid-cols-1 gap-6">
            {/* Performance Metrics Table */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Performance Metrics</h3>
              {modelInfo ? (
                <div className="overflow-x-auto">
                  <table className="min-w-full bg-white">
                    <thead>
                      <tr>
                        <th className="py-3 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Metric
                        </th>
                        <th className="py-3 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Value
                        </th>
                        <th className="py-3 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Visualization
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="py-4 px-4 border-b border-gray-200">
                          <div className="text-sm font-medium text-gray-900">Accuracy</div>
                          <div className="text-sm text-gray-500">Overall model accuracy</div>
                        </td>
                        <td className="py-4 px-4 border-b border-gray-200">
                          <div className="text-sm text-gray-900 font-medium">{(modelInfo.accuracy * 100).toFixed(2)}%</div>
                        </td>
                        <td className="py-4 px-4 border-b border-gray-200 w-1/3">
                          <div className="w-full bg-gray-200 rounded-full h-2.5">
                            <div 
                              className="bg-indigo-600 h-2.5 rounded-full" 
                              style={{ width: `${modelInfo.accuracy * 100}%` }}
                            ></div>
                          </div>
                        </td>
                      </tr>
                      {modelInfo.precision !== undefined && (
                        <tr>
                          <td className="py-4 px-4 border-b border-gray-200">
                            <div className="text-sm font-medium text-gray-900">Precision</div>
                            <div className="text-sm text-gray-500">Positive prediction accuracy</div>
                          </td>
                          <td className="py-4 px-4 border-b border-gray-200">
                            <div className="text-sm text-gray-900 font-medium">{(modelInfo.precision * 100).toFixed(2)}%</div>
                          </td>
                          <td className="py-4 px-4 border-b border-gray-200">
                            <div className="w-full bg-gray-200 rounded-full h-2.5">
                              <div 
                                className="bg-blue-500 h-2.5 rounded-full" 
                                style={{ width: `${modelInfo.precision * 100}%` }}
                              ></div>
                            </div>
                          </td>
                        </tr>
                      )}
                      {modelInfo.recall !== undefined && (
                        <tr>
                          <td className="py-4 px-4 border-b border-gray-200">
                            <div className="text-sm font-medium text-gray-900">Recall</div>
                            <div className="text-sm text-gray-500">Positive identification rate</div>
                          </td>
                          <td className="py-4 px-4 border-b border-gray-200">
                            <div className="text-sm text-gray-900 font-medium">{(modelInfo.recall * 100).toFixed(2)}%</div>
                          </td>
                          <td className="py-4 px-4 border-b border-gray-200">
                            <div className="w-full bg-gray-200 rounded-full h-2.5">
                              <div 
                                className="bg-green-500 h-2.5 rounded-full" 
                                style={{ width: `${modelInfo.recall * 100}%` }}
                              ></div>
                            </div>
                          </td>
                        </tr>
                      )}
                      {modelInfo.f1 !== undefined && (
                        <tr>
                          <td className="py-4 px-4 border-b border-gray-200">
                            <div className="text-sm font-medium text-gray-900">F1 Score</div>
                            <div className="text-sm text-gray-500">Balance of precision and recall</div>
                          </td>
                          <td className="py-4 px-4 border-b border-gray-200">
                            <div className="text-sm text-gray-900 font-medium">{(modelInfo.f1 * 100).toFixed(2)}%</div>
                          </td>
                          <td className="py-4 px-4 border-b border-gray-200">
                            <div className="w-full bg-gray-200 rounded-full h-2.5">
                              <div 
                                className="bg-purple-500 h-2.5 rounded-full" 
                                style={{ width: `${modelInfo.f1 * 100}%` }}
                              ></div>
                            </div>
                          </td>
                        </tr>
                      )}
                      {modelInfo.roc_auc !== undefined && (
                        <tr>
                          <td className="py-4 px-4 border-b border-gray-200">
                            <div className="text-sm font-medium text-gray-900">ROC AUC</div>
                            <div className="text-sm text-gray-500">Area under ROC curve</div>
                          </td>
                          <td className="py-4 px-4 border-b border-gray-200">
                            <div className="text-sm text-gray-900 font-medium">{(modelInfo.roc_auc * 100).toFixed(2)}%</div>
                          </td>
                          <td className="py-4 px-4 border-b border-gray-200">
                            <div className="w-full bg-gray-200 rounded-full h-2.5">
                              <div 
                                className="bg-yellow-500 h-2.5 rounded-full" 
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
                <div className="text-gray-500">No model metrics available</div>
              )}
            </div>

            {/* Confusion Matrix */}
            {renderConfusionMatrix()}
          </div>
        );

      case 'errors':
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Error Analysis */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Error Analysis</h3>
              {errorAnalysis ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 p-4 rounded">
                      <div className="text-sm text-gray-500">Error Count</div>
                      <div className="text-2xl font-bold">{errorAnalysis.error_count}</div>
                    </div>
                    <div className="bg-gray-50 p-4 rounded">
                      <div className="text-sm text-gray-500">Error Rate</div>
                      <div className="text-2xl font-bold">{(errorAnalysis.error_rate * 100).toFixed(2)}%</div>
                    </div>
                  </div>
                  
                  <div className="mt-4">
                    <div className="flex justify-between mb-1">
                      <span className="text-gray-600">Correct vs. Errors</span>
                    </div>
                    <ResponsiveContainer width="100%" height={200}>
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
              ) : (
                <div className="text-gray-500">Error analysis unavailable</div>
              )}
            </div>

            {/* Error Types */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Error Types</h3>
              {errorAnalysis && errorAnalysis.error_types ? (
                <div className="overflow-auto max-h-96">
                  <table className="min-w-full bg-white">
                    <thead>
                      <tr>
                        <th className="py-3 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Error Type
                        </th>
                        <th className="py-3 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Count
                        </th>
                        <th className="py-3 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Percentage
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {errorAnalysis.error_types.map((type, idx) => (
                        <tr key={idx}>
                          <td className="py-3 px-4 border-b border-gray-200">
                            <div className="text-sm font-medium text-gray-900">
                              {`Class ${type.true_class} → ${type.predicted_class}`}
                            </div>
                          </td>
                          <td className="py-3 px-4 border-b border-gray-200">
                            <div className="text-sm text-gray-900">{type.count}</div>
                          </td>
                          <td className="py-3 px-4 border-b border-gray-200">
                            <div className="text-sm text-gray-900">
                              {((type.count / errorAnalysis.error_count) * 100).toFixed(2)}%
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-gray-500">Error type analysis unavailable</div>
              )}
            </div>
          </div>
        );

      case 'features':
        return (
          <div className="grid grid-cols-1 gap-6">
            {/* Feature Importance */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Feature Importance</h3>
              {featureImportance ? (
                <div>
                  <div className="text-sm mb-4">
                    <span className="font-medium">Method: </span>
                    <span className="text-gray-700">{featureImportance.importance_method}</span>
                  </div>
                  <ResponsiveContainer width="100%" height={400}>
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
              ) : (
                <div className="text-gray-500">Feature importance unavailable</div>
              )}
            </div>
          </div>
        );

      case 'confidence':
        return (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Confidence Analysis */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Confidence Metrics</h3>
              {confidenceAnalysis ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 p-4 rounded">
                      <div className="text-sm text-gray-500">Average Confidence</div>
                      <div className="text-2xl font-bold">{(confidenceAnalysis.avg_confidence * 100).toFixed(2)}%</div>
                    </div>
                    <div className="bg-gray-50 p-4 rounded">
                      <div className="text-sm text-gray-500">Calibration Error</div>
                      <div className="text-2xl font-bold">{(confidenceAnalysis.calibration_error * 100).toFixed(2)}%</div>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 p-4 rounded">
                      <div className="text-sm text-gray-500">Avg. Correct Confidence</div>
                      <div className="text-2xl font-bold">{(confidenceAnalysis.avg_correct_confidence * 100).toFixed(2)}%</div>
                    </div>
                    <div className="bg-gray-50 p-4 rounded">
                      <div className="text-sm text-gray-500">Avg. Incorrect Confidence</div>
                      <div className="text-2xl font-bold">{(confidenceAnalysis.avg_incorrect_confidence * 100).toFixed(2)}%</div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-gray-500">Confidence analysis unavailable</div>
              )}
            </div>

            {/* Confidence Distribution */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Confidence Distribution</h3>
              {confidenceAnalysis && confidenceAnalysis.confidence_distribution ? (
                <ResponsiveContainer width="100%" height={300}>
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
              ) : (
                <div className="text-gray-500">Confidence distribution unavailable</div>
              )}
            </div>

            {/* Overconfident Examples */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Overconfident Predictions</h3>
              {confidenceAnalysis && confidenceAnalysis.overconfident_examples ? (
                <div>
                  <div className="mb-4">
                    <span className="text-gray-600">Threshold: </span>
                    <span className="font-medium">{confidenceAnalysis.overconfident_examples.threshold * 100}%</span>
                  </div>
                  <div className="mb-4">
                    <span className="text-gray-600">Count: </span>
                    <span className="font-medium">{confidenceAnalysis.overconfident_examples.count} examples</span>
                  </div>
                  <div className="p-3 bg-red-50 rounded-md border border-red-200">
                    <p className="text-sm text-red-800">
                      These are examples where the model was highly confident but wrong, indicating potential blind spots.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="text-gray-500">No overconfident examples data</div>
              )}
            </div>

            {/* Underconfident Examples */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Underconfident Predictions</h3>
              {confidenceAnalysis && confidenceAnalysis.underconfident_examples ? (
                <div>
                  <div className="mb-4">
                    <span className="text-gray-600">Threshold: </span>
                    <span className="font-medium">{confidenceAnalysis.underconfident_examples.threshold * 100}%</span>
                  </div>
                  <div className="mb-4">
                    <span className="text-gray-600">Count: </span>
                    <span className="font-medium">{confidenceAnalysis.underconfident_examples.count} examples</span>
                  </div>
                  <div className="p-3 bg-yellow-50 rounded-md border border-yellow-200">
                    <p className="text-sm text-yellow-800">
                      These are examples where the model was correct but had low confidence, indicating room for improved calibration.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="text-gray-500">No underconfident examples data</div>
              )}
            </div>
          </div>
        );

      case 'predictions':
        return (
          <div className="grid grid-cols-1 gap-6">
            {/* Sample Predictions - Mocked for now, would connect to API in real implementation */}
            <div className="bg-white p-6 rounded-lg shadow">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold">Sample Predictions</h3>
                <button className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700">
                  Fetch Samples
                </button>
              </div>
              
              <div className="flex space-x-4 mb-4">
                <div className="flex items-center">
                  <input type="checkbox" id="errors-only" className="mr-2" />
                  <label htmlFor="errors-only" className="text-sm text-gray-700">Show errors only</label>
                </div>
                
                <div className="flex items-center">
                  <label htmlFor="limit" className="text-sm text-gray-700 mr-2">Limit:</label>
                  <select id="limit" className="border rounded px-2 py-1 text-sm">
                    <option>10</option>
                    <option>20</option>
                    <option>50</option>
                    <option>100</option>
                  </select>
                </div>
              </div>
              
              <div className="overflow-x-auto">
                <table className="min-w-full bg-white">
                  <thead>
                    <tr>
                      <th className="py-3 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Sample ID
                      </th>
                      <th className="py-3 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        True Label
                      </th>
                      <th className="py-3 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Predicted
                      </th>
                      <th className="py-3 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Confidence
                      </th>
                      <th className="py-3 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Status
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {/* Demo data - would be replaced with real API data */}
                    {[...Array(10)].map((_, idx) => {
                      const isError = idx % 3 === 0;
                      return (
                        <tr key={idx} className={isError ? "bg-red-50" : "hover:bg-gray-50"}>
                          <td className="py-3 px-4 border-b border-gray-200">
                            <div className="text-sm font-medium text-gray-900">#{idx + 1}</div>
                          </td>
                          <td className="py-3 px-4 border-b border-gray-200">
                            <div className="text-sm text-gray-900">{idx % 2}</div>
                          </td>
                          <td className="py-3 px-4 border-b border-gray-200">
                            <div className="text-sm text-gray-900">{isError ? (idx % 2 === 0 ? 1 : 0) : idx % 2}</div>
                          </td>
                          <td className="py-3 px-4 border-b border-gray-200">
                            <div className="text-sm text-gray-900">
                              {isError ? `${(0.7 + Math.random() * 0.2).toFixed(2)}` : `${(0.8 + Math.random() * 0.15).toFixed(2)}`}
                            </div>
                          </td>
                          <td className="py-3 px-4 border-b border-gray-200">
                            <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${isError ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}`}>
                              {isError ? 'Error' : 'Correct'}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              
              <div className="flex justify-between items-center mt-4">
                <div className="text-sm text-gray-700">
                  Showing <span className="font-medium">1</span> to <span className="font-medium">10</span> of <span className="font-medium">100</span> results
                </div>
                <div className="flex space-x-2">
                  <button className="px-3 py-1 border rounded text-sm disabled:opacity-50">Previous</button>
                  <button className="px-3 py-1 border rounded text-sm bg-indigo-50">Next</button>
                </div>
              </div>
            </div>
          </div>
        );

      case 'training':
        return (
          <div className="grid grid-cols-1 gap-6">
            {/* Training History */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Training History</h3>
              {trainingHistory && trainingHistory.length > 0 ? (
                <ResponsiveContainer width="100%" height={400}>
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
              ) : (
                <div className="text-gray-500">No training history available</div>
              )}
            </div>
            
            {/* Learning Rate */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-semibold mb-4">Learning Rate</h3>
              {trainingHistory && trainingHistory.length > 0 && trainingHistory[0].learning_rate ? (
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={trainingHistory}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="iteration" />
                    <YAxis />
                    <Tooltip formatter={(value) => [value.toExponential(4), 'Learning Rate']} />
                    <Line type="monotone" dataKey="learning_rate" stroke="#ff7300" name="Learning Rate" />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-gray-500">No learning rate data available</div>
              )}
            </div>
          </div>
        );

      default:
        return <div>Select a tab to view data</div>;
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">CompileML Dashboard</h1>
              <p className="text-gray-500 mt-1">
                {modelInfo 
                  ? `Connected to ${modelInfo.name} (${modelInfo.framework})`
                  : 'No model connected'}
              </p>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${serverStatus?.status === 'online' ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm font-medium">
                {serverStatus?.status === 'online' ? 'Server Online' : 'Server Offline'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderTabs()}
        {renderTabContent()}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div className="text-sm text-gray-500">
              CompileML Dashboard v1.0.0
            </div>
            <div className="text-sm text-gray-500">
              {serverStatus?.started_at && `Server started: ${new Date(serverStatus.started_at).toLocaleString()}`}
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default CompileMLDashboard;