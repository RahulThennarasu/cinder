import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

export const getModelInfo = async () => {
  const response = await axios.get(`${API_URL}/model`);
  return response.data;
};

export const getErrorAnalysis = async () => {
  const response = await axios.get(`${API_URL}/errors`);
  return response.data;
};