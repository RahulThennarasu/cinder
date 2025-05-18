import axios from 'axios';

// Default API URL - can be configured
let API_URL = 'http://localhost:8000/api';

// Configure the API URL
export const configureAPI = (baseUrl) => {
  API_URL = baseUrl.endsWith('/api') ? baseUrl : `${baseUrl}/api`;
  console.log(`API URL configured to: ${API_URL}`);
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
  console.log(`Fetching model info from ${API_URL}/model`);
  return handleResponse(() => apiClient.get(`${API_URL}/model`));
};

export const getErrorAnalysis = async () => {
  console.log(`Fetching error analysis from ${API_URL}/errors`);
  return handleResponse(() => apiClient.get(`${API_URL}/errors`));
};

export const getTrainingHistory = async () => {
  console.log(`Fetching training history from ${API_URL}/training-history`);
  return handleResponse(() => apiClient.get(`${API_URL}/training-history`));
};

export const getErrorTypes = async () => {
  console.log(`Fetching error types from ${API_URL}/error-types`);
  return handleResponse(() => apiClient.get(`${API_URL}/error-types`));
};

export const getConfusionMatrix = async () => {
  console.log(`Fetching confusion matrix from ${API_URL}/confusion-matrix`);
  return handleResponse(() => apiClient.get(`${API_URL}/confusion-matrix`));
};

export const getPredictionDistribution = async () => {
  console.log(`Fetching prediction distribution from ${API_URL}/prediction-distribution`);
  return handleResponse(() => apiClient.get(`${API_URL}/prediction-distribution`));
};

export const getSamplePredictions = async (limit = 10, offset = 0) => {
  console.log(`Fetching sample predictions from ${API_URL}/sample-predictions`);
  return handleResponse(() => 
    apiClient.get(`${API_URL}/sample-predictions`, {
      params: { limit, offset }
    })
  );
};

export const getServerStatus = async () => {
  console.log(`Checking server status at ${API_URL}/status`);
  return handleResponse(() => apiClient.get(`${API_URL}/status`));
};

// Function for testing the connection
export const testConnection = async (url) => {
  const testUrl = url.endsWith('/') ? url : `${url}/`;
  console.log(`Testing connection to ${testUrl}`);
  
  try {
    const response = await axios.get(testUrl);
    console.log('Connection successful:', response.data);
    return { 
      success: true, 
      message: response.data.message || 'Connected successfully', 
      status: response.status 
    };
  } catch (error) {
    console.error('Connection failed:', error.message);
    return { 
      success: false, 
      message: error.message, 
      status: error.response?.status || 0 
    };
  }
};