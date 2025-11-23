import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add interceptor to include token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

const apiService = {
  // Auth
  login: (email, password) => {
    const formData = new FormData();
    formData.append('username', email);
    formData.append('password', password);
    return api.post('/auth/login', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
  },
  signup: (email, password, fullName) =>
    api.post('/auth/signup', { email, password, full_name: fullName }),
  getCurrentUser: () => api.get('/auth/me'),

  // Data
  getProducts: () => api.get('/data/products'),
  getDashboardMetrics: () => api.get('/dashboard/metrics'),

  // Forecasts
  getDemandForecasts: (productIds) =>
    api.get('/forecasts/demand', { params: { product_ids: productIds } }),
  createForecast: (productId, days, showCI) =>
    api.post('/forecasts/generate', { product_id: productId, days, show_ci: showCI }),

  // Inventory
  optimizeInventory: (productId) =>
    api.get(`/inventory/optimization`, { params: { product_id: productId } }),

  // Routes
  optimizeRoutes: (date, vehicles) =>
    api.get('/routes/clusters', { params: { date, clusters: vehicles } }),

  // Anomalies
  detectAnomalies: (productId, days) =>
    api.get('/anomalies/detect', { params: { product_id: productId, days } }),

  // Load Data
  loadData: () => api.post('/data/load'),
};

export default apiService;
