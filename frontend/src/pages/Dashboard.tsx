import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { FiActivity, FiUsers, FiBarChart2, FiAlertCircle } from 'react-icons/fi';
import { Card, LoadingSpinner, ErrorMessage } from '@/components/common';
import { healthApi, modelApi, dataApi } from '@/services/api';
import type { HealthResponse, ModelMetrics, DataStats } from '@/types/api';

const Dashboard: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelMetrics[]>([]);
  const [dataStats, setDataStats] = useState<DataStats | null>(null);
  const [loading, setLoading] = useState(false); // Changed to false - show content immediately
  const [error, setError] = useState<string | null>(null);
  const [isLoadingAPI, setIsLoadingAPI] = useState(true);

  useEffect(() => {
    // Load data in background, don't block UI
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setIsLoadingAPI(true);
      setError(null);

      const [healthData, modelsData, statsData] = await Promise.all([
        healthApi.check(),
        modelApi.listModels(),
        dataApi.getStats(),
      ]);

      setHealth(healthData);
      setModels(modelsData);
      setDataStats(statsData);
    } catch (err) {
      setError('Backend API not connected. The UI is working but predictions require the backend server.');
      console.error('Dashboard error:', err);
    } finally {
      setIsLoadingAPI(false);
    }
  };

  // Show API status banner if there's an error
  const showAPIWarning = error && !health;

  if (error && !health && !models.length) {
    // Only show full error state if we have no data at all
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
            Dashboard
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Overview of the diabetes prediction system
          </p>
        </div>
        <div className="max-w-2xl">
          <ErrorMessage
            title="Connection Error"
            message={error + " The frontend is running but cannot reach the backend API. Please make sure the backend server is running on http://localhost:8000"}
            variant="error"
          />
          <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
            <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-2">
              Quick Fix:
            </h3>
            <p className="text-blue-800 dark:text-blue-200 mb-2">
              Start the backend server in a terminal:
            </p>
            <code className="block p-3 bg-gray-800 text-green-400 rounded font-mono text-sm">
              cd backend<br/>
              venv\Scripts\python.exe app.py
            </code>
            <button
              onClick={loadDashboardData}
              className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
            >
              Retry Connection
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* API Status Banner */}
      {isLoadingAPI && (
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
          <div className="flex items-center gap-3">
            <LoadingSpinner size="sm" />
            <span className="text-blue-900 dark:text-blue-100">
              Connecting to backend API...
            </span>
          </div>
        </div>
      )}

      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
          Dashboard
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Overview of the diabetes prediction system
        </p>
      </div>

      {/* System Status */}
      {health && (
        <Card>
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold mb-2">System Status</h3>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-success-500 rounded-full animate-pulse" />
                <span className="text-success-600 dark:text-success-400 font-medium">
                  {health.status}
                </span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-500 dark:text-gray-400">
                Models Loaded
              </div>
              <div className="text-2xl font-bold text-primary-600 dark:text-primary-400">
                {health.models_loaded}
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Link to="/predict">
          <Card hoverable className="h-full">
            <div className="flex items-start gap-4">
              <div className="p-3 bg-primary-100 dark:bg-primary-900/30 rounded-lg">
                <FiActivity className="w-6 h-6 text-primary-600 dark:text-primary-400" />
              </div>
              <div>
                <h3 className="font-semibold text-lg mb-1">Single Prediction</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Predict diabetes risk for a single patient with detailed explanations
                </p>
              </div>
            </div>
          </Card>
        </Link>

        <Link to="/batch">
          <Card hoverable className="h-full">
            <div className="flex items-start gap-4">
              <div className="p-3 bg-success-100 dark:bg-success-900/30 rounded-lg">
                <FiUsers className="w-6 h-6 text-success-600 dark:text-success-400" />
              </div>
              <div>
                <h3 className="font-semibold text-lg mb-1">Batch Analysis</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Analyze multiple patients at once for efficient processing
                </p>
              </div>
            </div>
          </Card>
        </Link>

        <Link to="/compare">
          <Card hoverable className="h-full">
            <div className="flex items-start gap-4">
              <div className="p-3 bg-warning-100 dark:bg-warning-900/30 rounded-lg">
                <FiBarChart2 className="w-6 h-6 text-warning-600 dark:text-warning-400" />
              </div>
              <div>
                <h3 className="font-semibold text-lg mb-1">Model Comparison</h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Compare predictions from different machine learning models
                </p>
              </div>
            </div>
          </Card>
        </Link>
      </div>

      {/* Models Overview */}
      {models.length > 0 && (
        <Card title="Available Models" subtitle="Performance metrics for each model">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
            {models.map((model) => (
              <div
                key={model.model_name}
                className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg"
              >
                <h4 className="font-semibold text-lg mb-3 capitalize">
                  {model.model_name.replace(/_/g, ' ')}
                </h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Accuracy:</span>
                    <span className="font-medium">
                      {(model.accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Precision:</span>
                    <span className="font-medium">
                      {(model.precision * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Recall:</span>
                    <span className="font-medium">
                      {(model.recall * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">F1 Score:</span>
                    <span className="font-medium">
                      {(model.f1_score * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Dataset Statistics */}
      {dataStats && (
        <Card title="Dataset Statistics" subtitle="Training data overview">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
            <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                Total Samples
              </div>
              <div className="text-2xl font-bold">
                {dataStats.total_samples}
              </div>
            </div>
            <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                Features
              </div>
              <div className="text-2xl font-bold">
                {dataStats.features_count}
              </div>
            </div>
            <div className="p-4 bg-success-50 dark:bg-success-900/20 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                No Diabetes
              </div>
              <div className="text-2xl font-bold text-success-600 dark:text-success-400">
                {dataStats.class_distribution['No Diabetes'] || 0}
              </div>
            </div>
            <div className="p-4 bg-danger-50 dark:bg-danger-900/20 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                Diabetes
              </div>
              <div className="text-2xl font-bold text-danger-600 dark:text-danger-400">
                {dataStats.class_distribution['Diabetes'] || 0}
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Info Banner */}
      <Card>
        <div className="flex items-start gap-3">
          <FiAlertCircle className="w-5 h-5 text-primary-600 dark:text-primary-400 mt-0.5" />
          <div>
            <h4 className="font-semibold mb-1">About This System</h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              This system uses three machine learning models (Decision Tree, Random Forest, and XGBoost)
              to predict diabetes risk based on patient health metrics. Each model provides predictions
              with confidence scores and detailed explanations using SHAP values.
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default Dashboard;
