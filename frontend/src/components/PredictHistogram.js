// src/components/PredictionHistogram.js
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// This component would display a histogram of prediction distributions
// It would require additional data from the backend
const PredictionHistogram = ({ histogramData = [], isLoading, error }) => {
  if (isLoading) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Prediction Distribution</h2>
        <p>Loading histogram data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Prediction Distribution</h2>
        <p className="text-red-500">Error: {error.message}</p>
      </div>
    );
  }

  if (!histogramData || histogramData.length === 0) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Prediction Distribution</h2>
        <p>No histogram data available.</p>
        <p className="text-sm text-gray-500 mt-2">
          Note: This feature requires additional data from the backend.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4">Prediction Distribution</h2>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={histogramData}
            margin={{
              top: 5,
              right: 30,
              left: 20,
              bottom: 5,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="class" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="count" fill="#82ca9d" name="Prediction Count" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default PredictionHistogram;
