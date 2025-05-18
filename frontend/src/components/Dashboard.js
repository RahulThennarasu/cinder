import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { getModelInfo, getErrorAnalysis, getTrainingHistory, getErrorTypes } from '../services/api';
import ClaudeLayout from '../components/ClaudeLayout';
import MLFlowChart from '../components/MLFlowChart';

// Sample data to demonstrate functionality
const INITIAL_MODEL_INFO = {
  name: "Test Model",
  framework: "pytorch",
  dataset_size: 1000,
  accuracy: 0.87
};

const INITIAL_ERROR_DATA = {
  error_count: 130,
  correct_count: 870,
  error_indices: Array.from({ length: 130 }, (_, i) => i * 7),
  error_types: [
    { name: "False Positives", value: 75 },
    { name: "False Negatives", value: 55 }
  ]
};

const COLORS = ['#9CDCFE', '#4EC9B0', '#DCDCAA', '#B5CEA8', '#C586C0', '#CE9178'];

const MLDashboard = () => {
  const [modelInfo, setModelInfo] = useState(INITIAL_MODEL_INFO);
  const [errorData, setErrorData] = useState(INITIAL_ERROR_DATA);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [errorTypes, setErrorTypes] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [connected, setConnected] = useState(false);
  const [serverUrl, setServerUrl] = useState("http://localhost:8000");
  const [activeSection, setActiveSection] = useState('dashboard'); // 'dashboard', 'flowchart', 'settings'
  
  // Simulate API call with demo data
  const fetchMockData = () => {
    setIsLoading(true);
    // In a real app, replace this with actual API calls
    setTimeout(() => {
      setModelInfo({
        ...INITIAL_MODEL_INFO,
        accuracy: Math.min(0.99, INITIAL_MODEL_INFO.accuracy + Math.random() * 0.05)
      });
      setErrorData({
        ...INITIAL_ERROR_DATA,
        error_count: Math.floor(Math.random() * 150) + 50,
        correct_count: 1000 - (Math.floor(Math.random() * 150) + 50)
      });
      // Generate mock training history
      const mockHistory = Array.from({ length: 10 }, (_, i) => ({
        iteration: i + 1,
        accuracy: 0.7 + (i * 0.02) + (Math.random() * 0.03)
      }));
      setTrainingHistory(mockHistory);
      
      // Generate mock error types
      setErrorTypes([
        { name: "False Positives", value: 75 },
        { name: "False Negatives", value: 55 }
      ]);
      
      setIsLoading(false);
    }, 1000);
  };
  
  // Real API call function
  const fetchAllData = async () => {
    setIsLoading(true);
    try {
      // Try to fetch from real API
      const modelInfoData = await getModelInfo();
      const errorAnalysisData = await getErrorAnalysis();
      const historyData = await getTrainingHistory();
      const typesData = await getErrorTypes();
      
      setModelInfo(modelInfoData);
      setErrorData(errorAnalysisData);
      setTrainingHistory(historyData);
      setErrorTypes(typesData);
      setError(null);
      setConnected(true);
    } catch (err) {
      console.error('Error fetching data:', err);
      setError(err);
      
      // If API fails, use mock data instead
      fetchMockData();
      
      // If we can't connect to the server, set connected to false
      setConnected(false);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    // Start with mock data
    fetchMockData();
  }, []);

  // Generate error distribution data from current error indices
  const generateErrorDistribution = () => {
    // Make sure errorData and error_indices exist before trying to use them
    if (!errorData || !errorData.error_indices || !Array.isArray(errorData.error_indices)) {
      return [];
    }
    
    const bucketSize = 10;
    const distribution = {};
    
    errorData.error_indices.forEach(index => {
      const bucketLabel = `${Math.floor(index / bucketSize) * bucketSize}-${Math.floor(index / bucketSize) * bucketSize + bucketSize - 1}`;
      distribution[bucketLabel] = (distribution[bucketLabel] || 0) + 1;
    });
    
    return Object.keys(distribution).map(key => ({
      range: key,
      count: distribution[key]
    })).sort((a, b) => {
      const aStart = parseInt(a.range.split('-')[0]);
      const bStart = parseInt(b.range.split('-')[0]);
      return aStart - bStart;
    });
  };

  const renderDashboardSection = () => (
    <>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
        {/* Model Info Card */}
        <div className="claude-card">
          <div className="claude-header">
            <h2 className="claude-title">Model Information</h2>
            {modelInfo && modelInfo.framework && (
              <span className="px-2 py-1 text-xs rounded-full" 
                    style={{ backgroundColor: '#4EC9B0' + '30', color: '#4EC9B0' }}>
                {modelInfo.framework}
              </span>
            )}
          </div>
          <div className="claude-card-body space-y-4">
            <div>
              <p className="text-claude-text-secondary text-sm">Name</p>
              <p className="text-claude-text-primary text-lg font-medium">{modelInfo?.name || "Unknown"}</p>
            </div>
            <div>
              <p className="text-claude-text-secondary text-sm">Dataset Size</p>
              <p className="text-claude-text-primary text-lg font-medium">
                {modelInfo?.dataset_size ? modelInfo.dataset_size.toLocaleString() : "0"} samples
              </p>
            </div>
            <div>
              <p className="text-claude-text-secondary text-sm">Accuracy</p>
              <div className="flex items-center">
                <div className="w-full bg-claude-highlight rounded-full h-3 mr-2">
                  <div 
                    className={`h-3 rounded-full`}
                    style={{ 
                      width: `${(modelInfo?.accuracy || 0) * 100}%`,
                      backgroundColor: (modelInfo?.accuracy || 0) >= 0.9 ? '#B5CEA8' : 
                                       (modelInfo?.accuracy || 0) >= 0.7 ? '#DCDCAA' : 
                                       '#CE9178'
                    }}
                  ></div>
                </div>
                <span className="text-lg font-medium text-claude-text-primary">
                  {((modelInfo?.accuracy || 0) * 100).toFixed(2)}%
                </span>
              </div>
            </div>
          </div>
        </div>
        
        {/* Error Distribution Chart */}
        <div className="claude-card">
          <div className="claude-header">
            <h2 className="claude-title">Error Distribution</h2>
          </div>
          <div className="claude-card-body h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={generateErrorDistribution()}
                margin={{ top: 5, right: 20, left: 0, bottom: 25 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#3E3E42" />
                <XAxis 
                  dataKey="range" 
                  angle={-45} 
                  textAnchor="end"
                  height={60}
                  tick={{ fontSize: 10, fill: '#9CA3AF' }}
                  stroke="#3E3E42"
                />
                <YAxis tick={{ fill: '#9CA3AF' }} stroke="#3E3E42" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#252526', 
                    border: '1px solid #3E3E42',
                    borderRadius: '0.375rem',
                    color: '#CCCCCC'
                  }}
                  itemStyle={{ color: '#CCCCCC' }}
                  labelStyle={{ color: '#9CA3AF' }}
                />
                <Bar dataKey="count" fill="#9CDCFE" name="Error Count" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        {/* Error Types Pie Chart */}
        <div className="claude-card">
          <div className="claude-header">
            <h2 className="claude-title">Error Types</h2>
          </div>
          <div className="claude-card-body h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={errorData?.error_types || []}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="value"
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  labelLine={{ stroke: '#3E3E42' }}
                >
                  {(errorData?.error_types || []).map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#252526', 
                    border: '1px solid #3E3E42',
                    borderRadius: '0.375rem',
                    color: '#CCCCCC'
                  }}
                  itemStyle={{ color: '#CCCCCC' }}
                  labelStyle={{ color: '#9CA3AF' }}
                  formatter={(value) => [`${value} errors`, "Count"]} 
                />
                <Legend 
                  verticalAlign="bottom" 
                  align="center" 
                  wrapperStyle={{ color: '#9CA3AF' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Error Analysis */}
        <div className="claude-card">
          <div className="claude-header">
            <h2 className="claude-title">Error Analysis</h2>
          </div>
          <div className="claude-card-body">
            <div className="mb-4">
              <div className="flex justify-between mb-2">
                <span className="text-claude-text-secondary">Error Rate</span>
                <span className="text-claude-text-primary font-medium">
                  {errorData ? 
                    ((errorData.error_count / (errorData.error_count + errorData.correct_count)) * 100).toFixed(2) : 
                    "0.00"}%
                </span>
              </div>
              <div className="w-full bg-claude-highlight rounded-full h-3">
                <div 
                  className="bg-syntax-string h-3 rounded-full"
                  style={{ width: errorData ? 
                    `${(errorData.error_count / (errorData.error_count + errorData.correct_count)) * 100}%` : 
                    "0%" }}
                ></div>
              </div>
            </div>
            
            <div className="mt-6">
              <h3 className="text-claude-text-primary font-medium mb-2">Error Indices Sample</h3>
              <div className="border border-claude-border rounded p-3 h-40 overflow-y-auto">
                <div className="grid grid-cols-5 gap-2">
                  {(errorData?.error_indices || []).slice(0, 50).map((index, i) => (
                    <div key={i} className="bg-claude-highlight p-1 text-center rounded text-sm text-claude-text-secondary">
                      {index}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Training History Chart */}
        <div className="claude-card">
          <div className="claude-header">
            <h2 className="claude-title">Training History</h2>
          </div>
          <div className="claude-card-body h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={trainingHistory || []}
                margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#3E3E42" />
                <XAxis 
                  dataKey="iteration" 
                  stroke="#3E3E42"
                  tick={{ fill: '#9CA3AF' }}
                />
                <YAxis 
                  domain={[0.6, 1]} 
                  stroke="#3E3E42"
                  tick={{ fill: '#9CA3AF' }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#252526', 
                    border: '1px solid #3E3E42',
                    borderRadius: '0.375rem',
                    color: '#CCCCCC'
                  }}
                  itemStyle={{ color: '#CCCCCC' }}
                  labelStyle={{ color: '#9CA3AF' }}
                />
                <Legend 
                  wrapperStyle={{ color: '#9CA3AF' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="accuracy" 
                  stroke="#4EC9B0" 
                  activeDot={{ r: 8 }} 
                  name="Accuracy"
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </>
  );

  const renderFlowchartSection = () => (
    <div className="mb-6">
      <div className="claude-card">
        <div className="claude-header">
          <h2 className="claude-title">Machine Learning Pipeline</h2>
        </div>
        <div className="claude-card-body p-0">
          <MLFlowChart />
        </div>
      </div>
    </div>
  );

  const renderSettingsSection = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="claude-card">
        <div className="claude-header">
          <h2 className="claude-title">Connection Settings</h2>
        </div>
        <div className="claude-card-body">
          <div className="mb-4">
            <label className="claude-label">Server URL</label>
            <input
              type="text"
              value={serverUrl}
              onChange={(e) => setServerUrl(e.target.value)}
              className="claude-input w-full"
            />
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-2 ${connected ? 'bg-syntax-class' : 'bg-syntax-string'}`}></div>
              <span className="text-claude-text-secondary text-sm">
                {connected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <button 
              onClick={() => {
                setIsLoading(true);
                // Attempt to connect to the server
                fetch(serverUrl)
                  .then(response => {
                    if (response.ok) {
                      setConnected(true);
                      fetchAllData();
                    } else {
                      setConnected(false);
                      setError(new Error(`Failed to connect to server: ${response.status}`));
                    }
                  })
                  .catch(err => {
                    console.error("Connection error:", err);
                    setConnected(false);
                    setError(err);
                    // Still use mock data even if connection fails
                    fetchMockData();
                  })
                  .finally(() => {
                    setIsLoading(false);
                  });
              }}
              className="claude-btn-primary"
              disabled={isLoading}
            >
              {isLoading ? 'Connecting...' : connected ? 'Reconnect' : 'Connect'}
            </button>
          </div>
          {error && (
            <div className="mt-4 p-3 bg-syntax-string bg-opacity-10 border-l-2 border-syntax-string rounded text-sm text-claude-text-primary">
              {error.message}
            </div>
          )}
        </div>
      </div>
      
      <div className="claude-card">
        <div className="claude-header">
          <h2 className="claude-title">Theme Settings</h2>
        </div>
        <div className="claude-card-body">
          <p className="text-claude-text-secondary mb-4">
            This dashboard uses a Claude-inspired dark theme with VS Code syntax colors for a familiar development experience.
          </p>
          
          <div className="grid grid-cols-5 gap-2 mb-4">
            {COLORS.map((color, index) => (
              <div 
                key={index} 
                className="h-8 rounded flex items-center justify-center text-xs text-claude-bg font-mono"
                style={{ backgroundColor: color }}
              >
                {color}
              </div>
            ))}
          </div>
          
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-claude-text-secondary">Auto-refresh data</span>
              <button className="w-10 h-5 rounded-full bg-claude-highlight relative">
                <div className="absolute left-1 top-1 w-3 h-3 rounded-full bg-claude-text-primary"></div>
              </button>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-claude-text-secondary">Show animations</span>
              <button className="w-10 h-5 rounded-full bg-accent-primary relative">
                <div className="absolute right-1 top-1 w-3 h-3 rounded-full bg-white"></div>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <ClaudeLayout>
      {/* Navigation Tabs */}
      <div className="flex border-b border-claude-border mb-6">
        <button
          className={`px-4 py-3 text-sm font-medium ${
            activeSection === 'dashboard'
              ? 'text-accent-primary border-b-2 border-accent-primary'
              : 'text-claude-text-secondary hover:text-claude-text-primary'
          }`}
          onClick={() => setActiveSection('dashboard')}
        >
          Dashboard
        </button>
        <button
          className={`px-4 py-3 text-sm font-medium ${
            activeSection === 'flowchart'
              ? 'text-accent-primary border-b-2 border-accent-primary'
              : 'text-claude-text-secondary hover:text-claude-text-primary'
          }`}
          onClick={() => setActiveSection('flowchart')}
        >
          ML Pipeline
        </button>
        <button
          className={`px-4 py-3 text-sm font-medium ${
            activeSection === 'settings'
              ? 'text-accent-primary border-b-2 border-accent-primary'
              : 'text-claude-text-secondary hover:text-claude-text-primary'
          }`}
          onClick={() => setActiveSection('settings')}
        >
          Settings
        </button>
      </div>
      
      {/* Refresh and Status Bar */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center">
          <div className={`w-3 h-3 rounded-full mr-2 ${connected ? 'bg-syntax-class' : 'bg-syntax-string'}`}></div>
          <span className="text-claude-text-secondary text-sm">
            {connected ? 'Connected to server' : 'Using demo data (not connected)'}
          </span>
        </div>
        <button 
          onClick={fetchAllData}
          className="claude-btn-primary flex items-center"
          disabled={isLoading}
        >
          {isLoading ? (
            <>
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Refreshing...
            </>
          ) : (
            <>
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
              </svg>
              Refresh Data
            </>
          )}
        </button>
      </div>
      
      {/* Active Section Content */}
      {activeSection === 'dashboard' && renderDashboardSection()}
      {activeSection === 'flowchart' && renderFlowchartSection()}
      {activeSection === 'settings' && renderSettingsSection()}
    </ClaudeLayout>
  );
};

export default MLDashboard;