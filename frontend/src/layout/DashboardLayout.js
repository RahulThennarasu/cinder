import React, { useState } from 'react';
import Header from '../components/Header';
import ModelInfoCard from '../components/ModelInfoCard';
import ErrorAnalysis from '../components/ErrorAnalysis';
import ErrorDistribution from '../components/ErrorDistribution';
import AccuracyGauge from '../components/AccuracyGauge';
import ModelDetails from '../components/ModelDetails';
import ConnectionSettings from '../components/ConnectionSettings';
import { configureAPI, testConnection } from '../services/api';

const DashboardLayout = ({ 
  modelInfo, 
  errorData, 
  loading, 
  error, 
  refreshData, 
  connectionStatus, 
  setConnectionStatus 
}) => {
  const [serverUrl, setServerUrl] = useState('http://localhost:8000');
  const [showSettings, setShowSettings] = useState(false);

  const handleConnect = async (url) => {
    setConnectionStatus('connecting');
    const result = await testConnection(url);
    
    if (result.success) {
      configureAPI(url);
      setServerUrl(url);
      setConnectionStatus('connected');
      refreshData();
    } else {
      setConnectionStatus('disconnected');
    }
    
    // Hide settings after connection attempt
    setShowSettings(false);
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <Header />
      
      <div className="container mx-auto p-4">
        {/* Connection status and buttons */}
        <div className="flex justify-between items-center mb-6">
          <div className="flex items-center">
            <div className={`h-3 w-3 rounded-full mr-2 ${
              connectionStatus === 'connected' ? 'bg-green-500' : 
              connectionStatus === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
            }`}></div>
            <span className="font-medium">
              {connectionStatus === 'connected' ? 'Connected to model' : 
              connectionStatus === 'connecting' ? 'Connecting...' : 'Disconnected'}
            </span>
            {connectionStatus === 'connected' && (
              <span className="text-sm text-gray-500 ml-2">
                ({serverUrl})
              </span>
            )}
          </div>
          
          <div className="flex space-x-3">
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="bg-gray-200 hover:bg-gray-300 text-gray-800 px-4 py-2 rounded-md transition"
            >
              {showSettings ? 'Hide Settings' : 'Connection Settings'}
            </button>
            
            <button
              onClick={refreshData}
              className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-md transition"
              disabled={connectionStatus !== 'connected'}
            >
              Refresh Data
            </button>
          </div>
        </div>
        
        {/* Settings dialog */}
        {showSettings && (
          <div className="mb-6">
            <ConnectionSettings 
              onConnect={handleConnect} 
              isConnected={connectionStatus === 'connected'} 
              serverAddress={serverUrl} 
            />
          </div>
        )}
        
        {/* Main content grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-6 mb-6">
          {/* Top row */}
          <div className="col-span-1 md:col-span-2 lg:col-span-2">
            <ModelInfoCard 
              modelInfo={modelInfo} 
              isLoading={loading.modelInfo} 
              error={error.modelInfo} 
            />
          </div>
          <div className="col-span-1 md:col-span-1">
            <AccuracyGauge 
              accuracy={modelInfo?.accuracy || 0} 
              isLoading={loading.modelInfo} 
              error={error.modelInfo} 
            />
          </div>
          <div className="col-span-1 md:col-span-1">
            <ModelDetails 
              modelInfo={modelInfo} 
              isLoading={loading.modelInfo} 
              error={error.modelInfo} 
            />
          </div>
        </div>
        
        {/* Middle row */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <ErrorAnalysis 
            errorData={errorData} 
            isLoading={loading.errorData} 
            error={error.errorData} 
          />
          <ErrorDistribution 
            errorData={errorData} 
            isLoading={loading.errorData} 
            error={error.errorData} 
          />
        </div>
      </div>
    </div>
  );
};

export default DashboardLayout;