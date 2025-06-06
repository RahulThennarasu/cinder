import React, { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import ModelImprovementFlowchart from "./ModelImprovementFlowchart";
import EnhancedPredictionDistribution from "./EnhancedPredictionDistribution";
import ModelImprovementSuggestions from "./ModelImprovementSuggestions";
import BitAutoOptimizer from './BitAutoOptimizer';


// Constants
const COLORS = [
  "#D5451B",
  "#521C0D",
  "#FF9B45",
  "#ffd166",
  "#D5451B",
  "#82ca9d",
];

const DashboardContent = ({ serverStatus, modelInfo }) => {
  // State management
  const [confusionMatrix, setConfusionMatrix] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState(null);
  const [predictionDistribution, setPredictionDistribution] = useState(null);
  const [errorAnalysis, setErrorAnalysis] = useState(null);
  const [confidenceAnalysis, setConfidenceAnalysis] = useState(null);
  const [improvementSuggestions, setImprovementSuggestions] = useState(null);
  const [activeTab, setActiveTab] = useState("overview");

  // Fetch data from API
  useEffect(() => {
    const fetchData = async () => {
      const API_BASE_URL = "http://localhost:8000/api";

      try {
        // Only fetch if we have a server connection
        if (serverStatus && serverStatus.status === "online") {
          // Confusion matrix
          try {
            const confusionResponse = await fetch(
              `${API_BASE_URL}/confusion-matrix`,
            );
            const confusionData = await confusionResponse.json();
            setConfusionMatrix(confusionData);
          } catch (e) {
            console.error("Error fetching confusion matrix:", e);
          }

          // Feature importance
          try {
            const importanceResponse = await fetch(
              `${API_BASE_URL}/feature-importance`,
            );
            const importanceData = await importanceResponse.json();
            setFeatureImportance(importanceData);
          } catch (e) {
            console.error("Error fetching feature importance:", e);
          }

          // Training history
          try {
            const historyResponse = await fetch(
              `${API_BASE_URL}/training-history`,
            );
            const historyData = await historyResponse.json();
            setTrainingHistory(historyData);
          } catch (e) {
            console.error("Error fetching training history:", e);
          }

          // Prediction distribution
          try {
            const distributionResponse = await fetch(
              `${API_BASE_URL}/prediction-distribution`,
            );
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
            const confidenceResponse = await fetch(
              `${API_BASE_URL}/confidence-analysis`,
            );
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

    return featureImportance.feature_names
      .map((name, idx) => ({
        name: name,
        importance: featureImportance.importance_values[idx],
      }))
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 10);
  };

  // Prepare confidence distribution data
  const prepareConfidenceData = () => {
    if (!confidenceAnalysis || !confidenceAnalysis.confidence_distribution)
      return [];

    const { bin_edges, overall, correct, incorrect } =
      confidenceAnalysis.confidence_distribution;

    return bin_edges.slice(0, -1).map((edge, idx) => ({
      range: `${(edge * 100).toFixed(0)}-${(bin_edges[idx + 1] * 100).toFixed(0)}%`,
      overall: overall[idx],
      correct: correct[idx],
      incorrect: incorrect[idx],
    }));
  };

  // Helper function for confusion matrix visualization
  const renderConfusionMatrix = () => {
    if (!confusionMatrix)
      return (
        <div className="empty-state">Confusion matrix data unavailable</div>
      );

    return (
      <div className="card">
        <h3 className="card-title">Confusion Matrix</h3>
        <div className="table-container">
          <table className="table confusion-matrix">
            <thead>
              <tr>
                <th></th>
                {confusionMatrix.labels.map((label, idx) => (
                  <th key={idx}>Predicted: {label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {confusionMatrix.matrix.map((row, rowIdx) => (
                <tr key={rowIdx}>
                  <th>Actual: {confusionMatrix.labels[rowIdx]}</th>
                  {row.map((cell, cellIdx) => {
                    const isCorrect = rowIdx === cellIdx;
                    const opacity =
                      cell / Math.max(...confusionMatrix.matrix.flat());
                    const style = {
                      backgroundColor: isCorrect
                        ? `rgba(255, 123, 0, ${opacity})`
                        : `rgba(231, 76, 50, ${opacity * 0.7})`,
                    };

                    return (
                      <td
                        key={cellIdx}
                        className={
                          isCorrect ? "correct-cell" : "incorrect-cell"
                        }
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
      { id: "overview", label: "Overview" },
      { id: "performance", label: "Performance" },
      { id: "improvement", label: "Model Improvement" },
      { id: "improve", label: "Model Suggestions" },
      { id: "code", label: "Bit Model Design" },
    ];

    return (
      <div className="tabs">
        <nav className="tab-list">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`tab-button ${activeTab === tab.id ? "active" : ""}`}
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
      case "improvement":
        return (
          <div className="grid">
            <div className="card">
              <ModelImprovementFlowchart
                modelInfo={modelInfo}
                errorAnalysis={errorAnalysis}
                confidenceAnalysis={confidenceAnalysis}
              />
            </div>
          </div>
        );

      case "improve":
        return <ModelImprovementSuggestions />;

      case "overview":
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
                    <span className="info-value">
                      {modelInfo.dataset_size} samples
                    </span>
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
                    <span className="info-value status-value">
                      {serverStatus.status}
                    </span>
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
                    <span className="info-value">
                      {serverStatus.memory_usage?.toFixed(2)} MB
                    </span>
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
                      <span className="metric-value">
                        {(modelInfo.accuracy * 100).toFixed(2)}%
                      </span>
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
                        <span className="metric-value">
                          {(modelInfo.precision * 100).toFixed(2)}%
                        </span>
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
                        <span className="metric-value">
                          {(modelInfo.recall * 100).toFixed(2)}%
                        </span>
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
                        <span className="metric-value">
                          {(modelInfo.f1 * 100).toFixed(2)}%
                        </span>
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

            {/* Enhanced Prediction Distribution with Training History */}
            <div className="card" style={{ padding: 0 }}>
              <EnhancedPredictionDistribution
                predictionDistribution={predictionDistribution}
                errorAnalysis={errorAnalysis}
                confidenceAnalysis={confidenceAnalysis}
                featureImportance={featureImportance}
                modelInfo={modelInfo}
                trainingHistory={trainingHistory}
              />
            </div>
          </div>
        );

      case "performance":
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
                          <div className="table-sublabel">
                            Overall model accuracy
                          </div>
                        </td>
                        <td>
                          <div className="table-value">
                            {(modelInfo.accuracy * 100).toFixed(2)}%
                          </div>
                        </td>
                        <td style={{ width: "33%" }}>
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
                            <div className="table-sublabel">
                              Positive prediction accuracy
                            </div>
                          </td>
                          <td>
                            <div className="table-value">
                              {(modelInfo.precision * 100).toFixed(2)}%
                            </div>
                          </td>
                          <td>
                            <div className="progress-bar">
                              <div
                                className="progress-fill"
                                style={{
                                  width: `${modelInfo.precision * 100}%`,
                                }}
                              ></div>
                            </div>
                          </td>
                        </tr>
                      )}
                      {modelInfo.recall !== undefined && (
                        <tr>
                          <td>
                            <div className="table-label">Recall</div>
                            <div className="table-sublabel">
                              Positive identification rate
                            </div>
                          </td>
                          <td>
                            <div className="table-value">
                              {(modelInfo.recall * 100).toFixed(2)}%
                            </div>
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
                            <div className="table-sublabel">
                              Balance of precision and recall
                            </div>
                          </td>
                          <td>
                            <div className="table-value">
                              {(modelInfo.f1 * 100).toFixed(2)}%
                            </div>
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
                            <div className="table-sublabel">
                              Area under ROC curve
                            </div>
                          </td>
                          <td>
                            <div className="table-value">
                              {(modelInfo.roc_auc * 100).toFixed(2)}%
                            </div>
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

      case "code":
          return <BitAutoOptimizer modelInfo={modelInfo} />;
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
