// src/services/api.js
import axios from 'axios';

// Default API URL - can be configured
let API_URL = 'http://localhost:8000/api';

// Configure the API URL
export const configureAPI = (baseUrl) => {
  API_URL = baseUrl.endsWith('/api') ? baseUrl : `${baseUrl}/api`;
};

// Create axios instance with default config
const apiClient = axios.create({
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10 seconds timeout
});

// Handle API responses and errors consistently
const handleResponse = async (apiCall) => {
  try {
    const response = await apiCall();
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    
    // Format error for consistent handling
    const formattedError = {
      message: error.response?.data?.detail || error.message || 'Unknown error occurred',
      status: error.response?.status || 500,
      original: error
    };
    
    throw formattedError;
  }
};

// API endpoints
export const getModelInfo = async () => {
  return handleResponse(() => apiClient.get(`${API_URL}/model`));
};

export const getErrorAnalysis = async () => {
  return handleResponse(() => apiClient.get(`${API_URL}/errors`));
};

// Additional endpoints to implement on the backend later
export const getConfusionMatrix = async () => {
  return handleResponse(() => apiClient.get(`${API_URL}/confusion-matrix`));
};

export const getPredictionDistribution = async () => {
  return handleResponse(() => apiClient.get(`${API_URL}/prediction-distribution`));
};

export const getSamplePredictions = async (limit = 10, offset = 0) => {
  return handleResponse(() => 
    apiClient.get(`${API_URL}/sample-predictions`, {
      params: { limit, offset }
    })
  );
};

export const getServerStatus = async () => {
  return handleResponse(() => apiClient.get(`${API_URL}/status`));
};

// Function for testing the connection
export const testConnection = async (url) => {
  try {
    const response = await axios.get(url.endsWith('/') ? url : `${url}/`);
    return { 
      success: true, 
      message: response.data.message || 'Connected successfully', 
      status: response.status 
    };
  } catch (error) {
    return { 
      success: false, 
      message: error.message, 
      status: error.response?.status || 0 
    };
  }
};