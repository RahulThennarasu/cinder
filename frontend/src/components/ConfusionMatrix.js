// src/components/ConfusionMatrix.js
import React from 'react';

// This component would display a confusion matrix for classification models
// It would require additional data from the backend
const ConfusionMatrix = ({ matrix = null, labels = [], isLoading, error }) => {
  if (isLoading) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Confusion Matrix</h2>
        <p>Loading confusion matrix...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Confusion Matrix</h2>
        <p className="text-red-500">Error: {error.message}</p>
      </div>
    );
  }

  if (!matrix || matrix.length === 0 || labels.length === 0) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Confusion Matrix</h2>
        <p>No confusion matrix data available.</p>
        <p className="text-sm text-gray-500 mt-2">
          Note: This feature requires additional data from the backend.
        </p>
      </div>
    );
  }

  // Get the maximum value in the matrix for color scaling
  const maxValue = Math.max(...matrix.flat());

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4">Confusion Matrix</h2>
      
      <div className="overflow-x-auto">
        <table className="min-w-full">
          <thead>
            <tr>
              <th className="p-2 border-b-2 border-r-2"></th>
              {labels.map((label, idx) => (
                <th key={idx} className="p-2 border-b-2 text-center">
                  {label} (Predicted)
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rowIdx) => (
              <tr key={rowIdx}>
                <td className="p-2 border-r-2 font-medium">
                  {labels[rowIdx]} (Actual)
                </td>
                {row.map((cell, cellIdx) => {
                  // Calculate background color intensity based on value
                  const intensity = Math.round((cell / maxValue) * 100);
                  const isCorrect = rowIdx === cellIdx;
                  const bgColor = isCorrect
                    ? `rgba(0, 128, 0, ${intensity / 100})`
                    : `rgba(255, 0, 0, ${intensity / 100})`;
                  
                  return (
                    <td
                      key={cellIdx}
                      className="p-2 text-center border"
                      style={{ backgroundColor: bgColor }}
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

export default ConfusionMatrix;