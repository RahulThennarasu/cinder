// src/components/ErrorDistribution.js
import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const ErrorDistribution = ({ errorData, isLoading, error }) => {
  const [chartData, setChartData] = useState([]);

  useEffect(() => {
    if (errorData && errorData.error_indices) {
      // Group error indices into buckets for visualization
      const bucketSize = 10; // Group in buckets of 10
      const distribution = {};
      
      errorData.error_indices.forEach(index => {
        const bucketLabel = `${Math.floor(index / bucketSize) * bucketSize}-${Math.floor(index / bucketSize) * bucketSize + bucketSize - 1}`;
        distribution[bucketLabel] = (distribution[bucketLabel] || 0) + 1;
      });
      
      // Convert to format suitable for Recharts
      const data = Object.keys(distribution).map(key => ({
        range: key,
        count: distribution[key]
      }));
      
      // Sort by range
      data.sort((a, b) => {
        const aStart = parseInt(a.range.split('-')[0]);
        const bStart = parseInt(b.range.split('-')[0]);
        return aStart - bStart;
      });
      
      setChartData(data);
    } else {
      setChartData([]);
    }
  }, [errorData]);

  if (isLoading) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Error Distribution</h2>
        <p>Loading error distribution...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Error Distribution</h2>
        <p className="text-red-500">Error: {error.message}</p>
      </div>
    );
  }

  if (!errorData || chartData.length === 0) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Error Distribution</h2>
        <p>No error data available for visualization.</p>
      </div>
    );
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4">Error Distribution</h2>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            margin={{
              top: 5,
              right: 30,
              left: 20,
              bottom: 5,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="range" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="count" fill="#8884d8" name="Error Count" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ErrorDistribution;