import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { getModelInfo, getErrorAnalysis, getTrainingHistory, getErrorTypes } from '../services/api';

// VSCode theme colors
const COLORS = {
  BLUE: '#569CD6',
  LIGHT_BLUE: '#9CDCFE',
  GREEN: '#4EC9B0',
  YELLOW: '#DCDCAA',
  ORANGE: '#CE9178',
  PURPLE: '#C586C0',
  RED: '#F44747',
  GRAY: '#6B7280',
};

// Sample colors for pie chart and other visualizations
const CHART_COLORS = [COLORS.LIGHT_BLUE, COLORS.GREEN, COLORS.YELLOW, COLORS.ORANGE, COLORS.PURPLE];

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

const MLDashboard = () => {
  const [modelInfo, setModelInfo] = useState(INITIAL_MODEL_INFO);
  const [errorData, setErrorData] = useState(INITIAL_ERROR_DATA);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [errorTypes, setErrorTypes] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [connected, setConnected] = useState(false);
  const [serverUrl, setServerUrl] = useState("http://localhost:8000");
  const [activeTab, setActiveTab] = useState('dashboard'); // 'dashboard', 'settings'
  
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

  const renderDashboardTab = () => (
    <>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
        {/* Model Info Card */}
        <div className="bg-[#252526] border border-[#3E3E42] rounded-lg shadow-md overflow-hidden">
          <div className="px-4 py-3 border-b border-[#3E3E42] flex justify-between items-center">
            <h2 className="text-lg font-medium text-[#CCCCCC]">Model Information</h2>
            {modelInfo && modelInfo.framework && (
              <span className="px-2 py-1 text-xs rounded-full bg-opacity-20" 
                    style={{ backgroundColor: 'rgba(78, 201, 176, 0.2)', color: COLORS.GREEN }}>
                {modelInfo.framework}
              </span>
            )}
          </div>
          <div className="p-4 space-y-4">
            <div>
              <p className="text-[#9CA3AF] text-sm">Name</p>
              <p className="text-[#CCCCCC] text-lg font-medium">{modelInfo?.name || "Unknown"}</p>
            </div>
            <div>
              <p className="text-[#9CA3AF] text-sm">Dataset Size</p>
              <p className="text-[#CCCCCC] text-lg font-medium">
                {modelInfo?.dataset_size ? modelInfo.dataset_size.toLocaleString() : "0"} samples
              </p>
            </div>
            <div>
              <p className="text-[#9CA3AF] text-sm">Accuracy</p>
              <div className="flex items-center">
                <div className="w-full bg-[#2D2D30] rounded-full h-3 mr-2">
                  <div 
                    className="h-3 rounded-full transition-all duration-500"
                    style={{ 
                      width: `${(modelInfo?.accuracy || 0) * 100}%`,
                      backgroundColor: (modelInfo?.accuracy || 0) >= 0.9 ? COLORS.GREEN : 
                                      (modelInfo?.accuracy || 0) >= 0.7 ? COLORS.YELLOW : 
                                      COLORS.ORANGE
                    }}
                  ></div>
                </div>
                <span className="text-lg font-medium text-[#CCCCCC]">
                  {((modelInfo?.accuracy || 0) * 100).toFixed(2)}%
                </span>
              </div>
            </div>
          </div>
        </div>
        
        {/* Error Distribution Chart */}
        <div className="bg-[#252526] border border-[#3E3E42] rounded-lg shadow-md overflow-hidden">
          <div className="px-4 py-3 border-b border-[#3E3E42]">
            <h2 className="text-lg font-medium text-[#CCCCCC]">Error Distribution</h2>
          </div>
          <div className="p-4 h-64">
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
                <Bar dataKey="count" fill={COLORS.LIGHT_BLUE} name="Error Count" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        {/* Error Types Pie Chart */}
        <div className="bg-[#252526] border border-[#3E3E42] rounded-lg shadow-md overflow-hidden">
          <div className="px-4 py-3 border-b border-[#3E3E42]">
            <h2 className="text-lg font-medium text-[#CCCCCC]">Error Types</h2>
          </div>
          <div className="p-4 h-64">
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
                    <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
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
        <div className="bg-[#252526] border border-[#3E3E42] rounded-lg shadow-md overflow-hidden">
          <div className="px-4 py-3 border-b border-[#3E3E42]">
            <h2 className="text-lg font-medium text-[#CCCCCC]">Error Analysis</h2>
          </div>
          <div className="p-4">
            <div className="mb-4">
              <div className="flex justify-between mb-2">
                <span className="text-[#9CA3AF]">Error Rate</span>
                <span className="text-[#CCCCCC] font-medium">
                  {errorData ? 
                    ((errorData.error_count / (errorData.error_count + errorData.correct_count)) * 100).toFixed(2) : 
                    "0.00"}%
                </span>
              </div>
              <div className="w-full bg-[#2D2D30] rounded-full h-3">
                <div 
                  className="bg-[#CE9178] h-3 rounded-full transition-all duration-500"
                  style={{ width: errorData ? 
                    `${(errorData.error_count / (errorData.error_count + errorData.correct_count)) * 100}%` : 
                    "0%" }}
                ></div>
              </div>
            </div>
            
            <div className="mt-6">
              <h3 className="text-[#CCCCCC] font-medium mb-2">Error Indices Sample</h3>
              <div className="border border-[#3E3E42] rounded p-3 h-40 overflow-y-auto">
                <div className="grid grid-cols-5 gap-2">
                  {(errorData?.error_indices || []).slice(0, 50).map((index, i) => (
                    <div key={i} className="bg-[#2D2D30] p-1 text-center rounded text-sm text-[#9CA3AF]">
                      {index}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Training History Chart */}
        <div className="bg-[#252526] border border-[#3E3E42] rounded-lg shadow-md overflow-hidden">
          <div className="px-4 py-3 border-b border-[#3E3E42]">
            <h2 className="text-lg font-medium text-[#CCCCCC]">Training History</h2>
          </div>
          <div className="p-4 h-64">
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
                  stroke={COLORS.GREEN} 
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

  const renderSettingsTab = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="bg-[#252526] border border-[#3E3E42] rounded-lg shadow-md overflow-hidden">
        <div className="px-4 py-3 border-b border-[#3E3E42]">
          <h2 className="text-lg font-medium text-[#CCCCCC]">Connection Settings</h2>
        </div>
        <div className="p-4">
          <div className="mb-4">
            <label className="block text-sm font-medium text-[#9CA3AF] mb-1">Server URL</label>
            <input
              type="text"
              value={serverUrl}
              onChange={(e) => setServerUrl(e.target.value)}
              className="w-full bg-[#1E1E1E] border border-[#3E3E42] rounded-md px-3 py-2 text-[#CCCCCC] focus:outline-none focus:ring-2 focus:ring-[#007ACC] focus:border-[#007ACC] transition-colors"
            />
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <div className={`w-3 h-3 rounded-full mr-2 ${connected ? 'bg-[#4EC9B0]' : 'bg-[#CE9178]'}`}></div>
              <span className="text-[#9CA3AF] text-sm">
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
              className="bg-[#007ACC] hover:bg-[#0066B3] text-white px-4 py-2 rounded transition-colors"
              disabled={isLoading}
            >
              {isLoading ? 'Connecting...' : connected ? 'Reconnect' : 'Connect'}
            </button>
          </div>
          {error && (
            <div className="mt-4 p-3 bg-[#CE9178] bg-opacity-10 border-l-2 border-[#CE9178] rounded text-sm text-[#CCCCCC]">
              {error.message}
            </div>
          )}
        </div>
      </div>
      
      <div className="bg-[#252526] border border-[#3E3E42] rounded-lg shadow-md overflow-hidden">
        <div className="px-4 py-3 border-b border-[#3E3E42]">
          <h2 className="text-lg font-medium text-[#CCCCCC]">Backend Information</h2>
        </div>
        <div className="p-4">
          <p className="text-[#9CA3AF] mb-4">
            This dashboard connects to the ML model debugging backend API. The backend provides analysis
            of machine learning models including accuracy metrics, error analysis, and training history.
          </p>
          
          <h3 className="text-[#CCCCCC] font-medium mb-2">Available Endpoints</h3>
          <ul className="space-y-2 text-[#9CA3AF] text-sm">
            <li className="flex items-center">
              <span className="inline-block w-3 h-3 mr-2 rounded-full bg-[#4EC9B0]"></span>
              <code className="bg-[#1E1E1E] px-2 py-1 rounded mr-2 text-[#9CDCFE]">/api/model</code>
              <span>Model information</span>
            </li>
            <li className="flex items-center">
              <span className="inline-block w-3 h-3 mr-2 rounded-full bg-[#CE9178]"></span>
              <code className="bg-[#1E1E1E] px-2 py-1 rounded mr-2 text-[#9CDCFE]">/api/errors</code>
              <span>Error analysis</span>
            </li>
            <li className="flex items-center">
              <span className="inline-block w-3 h-3 mr-2 rounded-full bg-[#9CDCFE]"></span>
              <code className="bg-[#1E1E1E] px-2 py-1 rounded mr-2 text-[#9CDCFE]">/api/training-history</code>
              <span>Training history</span>
            </li>
            <li className="flex items-center">
              <span className="inline-block w-3 h-3 mr-2 rounded-full bg-[#C586C0]"></span>
              <code className="bg-[#1E1E1E] px-2 py-1 rounded mr-2 text-[#9CDCFE]">/api/error-types</code>
              <span>Error type breakdown</span>
            </li>
          </ul>
          
          <div className="mt-4 p-3 border border-[#3E3E42] rounded bg-[#1E1E1E]">
            <h4 className="text-[#569CD6] text-sm font-medium mb-1">Server Status</h4>
            <div className="flex justify-between text-sm">
              <span className="text-[#9CA3AF]">Status:</span>
              <span className={`${connected ? 'text-[#4EC9B0]' : 'text-[#CE9178]'}`}>
                {connected ? 'Online' : 'Offline'}
              </span>
            </div>
            {connected && (
              <div className="flex justify-between text-sm mt-1">
                <span className="text-[#9CA3AF]">URL:</span>
                <span className="text-[#DCDCAA]">{serverUrl}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#1E1E1E] text-[#CCCCCC] p-6">
      {/* Navigation Tabs */}
      <div className="flex border-b border-[#3E3E42] mb-6">
        <button
          className={`px-4 py-3 text-sm font-medium ${
            activeTab === 'dashboard'
              ? 'text-[#569CD6] border-b-2 border-[#569CD6]'
              : 'text-[#CCCCCC] hover:text-white'
          }`}
          onClick={() => setActiveTab('dashboard')}
        >
          Dashboard
        </button>
        <button
          className={`px-4 py-3 text-sm font-medium ${
            activeTab === 'settings'
              ? 'text-[#569CD6] border-b-2 border-[#569CD6]'
              : 'text-[#CCCCCC] hover:text-white'
          }`}
          onClick={() => setActiveTab('settings')}
        >
          Settings
        </button>
      </div>
      
      {/* Refresh and Status Bar */}
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center">
          <div className={`w-3 h-3 rounded-full mr-2 ${connected ? 'bg-[#4EC9B0]' : 'bg-[#CE9178]'}`}></div>
          <span className="text-[#9CA3AF] text-sm">
            {connected ? 'Connected to server' : 'Using demo data (not connected)'}
          </span>
        </div>
        <button 
          onClick={fetchAllData}
          className="bg-[#007ACC] hover:bg-[#0066B3] text-white px-4 py-2 rounded flex items-center transition-colors"
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
      
      {/* Active Tab Content */}
      {activeTab === 'dashboard' ? renderDashboardTab() : renderSettingsTab()}
    </div>
  );
};

export default MLDashboard;