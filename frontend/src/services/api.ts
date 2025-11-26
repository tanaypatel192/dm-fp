/**
 * API Service Layer
 *
 * This module provides a centralized interface for all API calls to the backend.
 * Uses axios for HTTP requests with error handling and loading states.
 */

import axios, { AxiosError, AxiosInstance } from 'axios';
import type {
  PatientInput,
  PredictionOutput,
  BatchPatientInput,
  BatchPredictionOutput,
  ModelMetrics,
  FeatureImportance,
  ModelComparisonOutput,
  DataStats,
  HealthResponse,
  ComprehensivePredictionOutput,
  ErrorResponse,
} from '@/types/api';

// Create axios instance with base configuration
const api: AxiosInstance = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 30000, // 30 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('[API] Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`[API] Response:`, response.status, response.data);
    return response;
  },
  (error: AxiosError<ErrorResponse>) => {
    if (error.response) {
      // Server responded with error
      console.error('[API] Server error:', error.response.status, error.response.data);
    } else if (error.request) {
      // Request made but no response
      console.error('[API] No response:', error.request);
    } else {
      // Error setting up request
      console.error('[API] Request setup error:', error.message);
    }
    return Promise.reject(error);
  }
);

/**
 * Health Check API
 */
export const healthApi = {
  /**
   * Check API health status
   */
  check: async (): Promise<HealthResponse> => {
    const response = await api.get<HealthResponse>('/health');
    return response.data;
  },
};

/**
 * Prediction API
 */
export const predictionApi = {
  /**
   * Make a single prediction
   * @param patient - Patient data
   * @param modelName - Model to use (default: "xgboost")
   */
  predict: async (
    patient: PatientInput,
    modelName: string = 'xgboost'
  ): Promise<PredictionOutput> => {
    const response = await api.post<PredictionOutput>(
      '/api/predict',
      patient,
      {
        params: { model_name: modelName },
      }
    );
    return response.data;
  },

  /**
   * Make batch predictions
   * @param batchInput - Array of patient data
   * @param modelName - Model to use
   */
  predictBatch: async (
    batchInput: BatchPatientInput,
    modelName: string = 'xgboost'
  ): Promise<BatchPredictionOutput> => {
    const response = await api.post<BatchPredictionOutput>(
      '/api/predict-batch',
      batchInput,
      {
        params: { model_name: modelName },
      }
    );
    return response.data;
  },

  /**
   * Get comprehensive prediction with explanations
   * @param patient - Patient data
   */
  predictExplain: async (
    patient: PatientInput
  ): Promise<ComprehensivePredictionOutput> => {
    const response = await api.post<ComprehensivePredictionOutput>(
      '/api/predict-explain',
      patient
    );
    return response.data;
  },

  /**
   * Compare predictions from all models
   * @param patient - Patient data
   */
  compareModels: async (
    patient: PatientInput
  ): Promise<ModelComparisonOutput> => {
    const response = await api.post<ModelComparisonOutput>(
      '/api/compare-models',
      patient
    );
    return response.data;
  },
};

/**
 * Model Information API
 */
export const modelApi = {
  /**
   * List all available models with metrics
   */
  listModels: async (): Promise<ModelMetrics[]> => {
    const response = await api.get<ModelMetrics[]>('/api/models');
    return response.data;
  },

  /**
   * Get metrics for a specific model
   * @param modelName - Name of the model
   */
  getMetrics: async (modelName: string): Promise<ModelMetrics> => {
    const response = await api.get<ModelMetrics>(
      `/api/model/${modelName}/metrics`
    );
    return response.data;
  },

  /**
   * Get feature importance for a model
   * @param modelName - Name of the model
   * @param topN - Number of top features to return (default: 10)
   */
  getFeatureImportance: async (
    modelName: string,
    topN: number = 10
  ): Promise<FeatureImportance[]> => {
    const response = await api.get<FeatureImportance[]>(
      `/api/model/${modelName}/feature-importance`,
      {
        params: { top_n: topN },
      }
    );
    return response.data;
  },
};

/**
 * Data Statistics API
 */
export const dataApi = {
  /**
   * Get dataset statistics
   */
  getStats: async (): Promise<DataStats> => {
    const response = await api.get<DataStats>('/api/data-stats');
    return response.data;
  },
};

/**
 * Helper function to handle API errors
 */
export const handleApiError = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<ErrorResponse>;
    if (axiosError.response?.data?.error) {
      return axiosError.response.data.error;
    }
    if (axiosError.response?.data) {
      return JSON.stringify(axiosError.response.data);
    }
    if (axiosError.message) {
      return axiosError.message;
    }
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unknown error occurred';
};

// Export the axios instance for custom requests
export default api;
