// src/components/ConnectionSettings.js
import React, { useState } from 'react';

const ConnectionSettings = ({ onConnect, isConnected, serverAddress = 'http://localhost:8000' }) => {
  const [address, setAddress] = useState(serverAddress);

  const handleConnect = (e) => {
    e.preventDefault();
    onConnect(address);
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4">Connection Settings</h2>
      
      <form onSubmit={handleConnect}>
        <div className="mb-4">
          <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="server-address">
            Server Address
          </label>
          <input
            id="server-address"
            type="text"
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
            placeholder="http://localhost:8000"
          />
        </div>
        
        <div className="flex items-center justify-between">
          <button
            type="submit"
            className={`${
              isConnected ? 'bg-green-500 hover:bg-green-600' : 'bg-blue-500 hover:bg-blue-600'
            } text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline`}
          >
            {isConnected ? 'Connected' : 'Connect'}
          </button>
          
          {isConnected && (
            <span className="text-green-500 text-sm">
              ✓ Connected to server
            </span>
          )}
        </div>
      </form>
    </div>
  );
};

export default ConnectionSettings;