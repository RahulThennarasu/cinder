// src/components/Dashboard.js
import React, { useState, useEffect } from 'react';
import { getModelInfo, getErrorAnalysis } from '../services/api';
import ModelInfoCard from './ModelInfoCard';
import ErrorAnalysis from './ErrorAnalysis';
import './card.css';

const Dashboard = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [errorData, setErrorData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        // Fetch model info
        const modelData = await getModelInfo();
        setModelInfo(modelData);
        
        // Fetch error analysis
        const errorAnalysisData = await getErrorAnalysis();
        setErrorData(errorAnalysisData);
        
        setError(null);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError(err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, []);

  return (
    <div className="dashboard">
      <h1>CompileML Dashboard</h1>
      
      <div className="dashboard-content">
        <ModelInfoCard 
          modelInfo={modelInfo} 
          isLoading={isLoading} 
          error={error} 
        />
        
        <ErrorAnalysis 
          errorData={errorData} 
          isLoading={isLoading} 
          error={error} 
        />
      </div>
    </div>
  );
};

export default Dashboard;