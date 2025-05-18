// src/components/AccuracyGauge.js
import React from 'react';

const AccuracyGauge = ({ accuracy = 0, isLoading, error }) => {
  // Calculate the position for the gauge needle
  const needleRotation = accuracy * 180;
  
  // Determine color based on accuracy
  let color = 'text-red-500';
  if (accuracy >= 0.9) {
    color = 'text-green-500';
  } else if (accuracy >= 0.7) {
    color = 'text-yellow-500';
  } else if (accuracy >= 0.5) {
    color = 'text-orange-500';
  }

  if (isLoading) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Model Accuracy</h2>
        <p>Loading accuracy information...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Model Accuracy</h2>
        <p className="text-red-500">Error: {error.message}</p>
      </div>
    );
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4">Model Accuracy</h2>
      <div className="flex flex-col items-center">
        {/* Gauge visualization */}
        <div className="relative w-48 h-24 overflow-hidden mb-2">
          {/* Gauge background */}
          <div className="absolute w-48 h-48 rounded-full border-8 border-gray-200 bottom-0"></div>
          
          {/* Gauge fill */}
          <div 
            className="absolute w-48 h-48 rounded-full border-8 border-transparent border-t-blue-500 border-r-blue-500 border-l-blue-500 bottom-0"
            style={{ 
              transform: `rotate(${needleRotation}deg)`,
              transition: 'transform 1s ease-out'
            }}
          ></div>
          
          {/* Gauge center */}
          <div className="absolute w-4 h-4 bg-gray-800 rounded-full left-1/2 bottom-0 transform -translate-x-1/2"></div>
        </div>
        
        {/* Accuracy percentage */}
        <div className={`text-4xl font-bold ${color}`}>
          {(accuracy * 100).toFixed(1)}%
        </div>
      </div>
    </div>
  );
};

export default AccuracyGauge;