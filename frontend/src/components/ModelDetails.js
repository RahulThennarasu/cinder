// src/components/ModelDetails.js
import React from 'react';

const ModelDetails = ({ modelInfo, isLoading, error }) => {
  if (isLoading) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Model Details</h2>
        <p>Loading model details...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Model Details</h2>
        <p className="text-red-500">Error: {error.message}</p>
      </div>
    );
  }

  if (!modelInfo) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-bold mb-4">Model Details</h2>
        <p>No model connected. Please connect a model to see details.</p>
      </div>
    );
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4">Model Details</h2>
      <table className="min-w-full bg-white">
        <tbody>
          <tr className="border-b">
            <td className="px-4 py-2 font-medium">Name:</td>
            <td className="px-4 py-2">{modelInfo.name}</td>
          </tr>
          <tr className="border-b">
            <td className="px-4 py-2 font-medium">Framework:</td>
            <td className="px-4 py-2 capitalize">{modelInfo.framework}</td>
          </tr>
          <tr className="border-b">
            <td className="px-4 py-2 font-medium">Dataset Size:</td>
            <td className="px-4 py-2">{modelInfo.dataset_size} samples</td>
          </tr>
          <tr>
            <td className="px-4 py-2 font-medium">Accuracy:</td>
            <td className="px-4 py-2">{(modelInfo.accuracy * 100).toFixed(2)}%</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default ModelDetails;