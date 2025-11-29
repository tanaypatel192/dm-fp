import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  FiActivity,
  FiServer,
  FiCpu,
  FiDatabase,
  FiAlertCircle,
  FiCheckCircle,
  FiClock,
  FiTrendingUp,
  FiTrendingDown,
  FiZap,
  FiX,
  FiRefreshCw,
} from 'react-icons/fi';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Card } from '@/components/common';
import { healthApi, modelApi } from '@/services/api';
import type { HealthResponse, ModelMetrics } from '@/types/api';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// Metric types
interface APIMetric {
  timestamp: number;
  responseTime: number;
  success: boolean;
  endpoint: string;
}

interface ModelHealth {
  name: string;
  status: 'healthy' | 'degraded' | 'down';
  lastPrediction: number | null;
  avgConfidence: number;
  errorRate: number;
}

interface SystemMetrics {
  memoryUsage: number; // Percentage
  cpuUsage: number; // Percentage
  activeConnections: number;
  cacheHitRate: number; // Percentage
}

interface HistoricalData {
  date: string;
  predictions: number;
  avgAccuracy: number;
  errorRate: number;
}

interface ToastNotification {
  id: string;
  type: 'success' | 'warning' | 'error' | 'info';
  message: string;
  timestamp: number;
}

// Constants
const RESPONSE_TIME_THRESHOLD = 2000; // 2 seconds
const ERROR_RATE_THRESHOLD = 0.1; // 10%
const POLLING_INTERVAL = 5000; // 5 seconds
const MAX_API_METRICS = 50; // Keep last 50 requests
const MAX_TOASTS = 5;

const PerformanceMonitor: React.FC = () => {
  // State
  const [apiMetrics, setApiMetrics] = useState<APIMetric[]>([]);
  const [modelHealth, setModelHealth] = useState<ModelHealth[]>([]);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    memoryUsage: 0,
    cpuUsage: 0,
    activeConnections: 0,
    cacheHitRate: 0,
  });
  const [historicalData, setHistoricalData] = useState<HistoricalData[]>([]);
  const [toasts, setToasts] = useState<ToastNotification[]>([]);
  const [isPolling, setIsPolling] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [predictionsPerMinute, setPredictionsPerMinute] = useState(0);

  // Add toast notification
  const addToast = useCallback((type: ToastNotification['type'], message: string) => {
    const toast: ToastNotification = {
      id: `toast-${Date.now()}-${Math.random()}`,
      type,
      message,
      timestamp: Date.now(),
    };

    setToasts((prev) => {
      const newToasts = [toast, ...prev].slice(0, MAX_TOASTS);
      return newToasts;
    });

    // Auto-remove after 5 seconds
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== toast.id));
    }, 5000);
  }, []);

  // Remove toast
  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  // Track API call
  const trackAPICall = useCallback(
    (endpoint: string, responseTime: number, success: boolean) => {
      const metric: APIMetric = {
        timestamp: Date.now(),
        responseTime,
        success,
        endpoint,
      };

      setApiMetrics((prev) => {
        const updated = [...prev, metric].slice(-MAX_API_METRICS);
        return updated;
      });

      // Alert on slow response
      if (responseTime > RESPONSE_TIME_THRESHOLD) {
        addToast('warning', `Slow response detected: ${responseTime}ms on ${endpoint}`);
      }

      // Alert on error
      if (!success) {
        addToast('error', `API error on ${endpoint}`);
      }

      // Success notification (only for predictions)
      if (success && endpoint.includes('predict')) {
        addToast('success', 'Prediction completed successfully');
      }
    },
    [addToast]
  );

  // Fetch health data
  const fetchHealthData = useCallback(async () => {
    try {
      const startTime = Date.now();
      const health = await healthApi.check();
      const responseTime = Date.now() - startTime;

      trackAPICall('/health', responseTime, true);

      // Update model health
      const models = await modelApi.listModels();
      const modelHealthData: ModelHealth[] = models.map((model) => ({
        name: model.model_name,
        status: model.is_loaded ? 'healthy' : 'down',
        lastPrediction: Date.now() - Math.random() * 60000, // Mock: last minute
        avgConfidence: model.accuracy || 0,
        errorRate: Math.random() * 0.05, // Mock: 0-5%
      }));

      setModelHealth(modelHealthData);

      // Update system metrics (mock data - backend would provide real metrics)
      setSystemMetrics({
        memoryUsage: 45 + Math.random() * 20, // 45-65%
        cpuUsage: 30 + Math.random() * 30, // 30-60%
        activeConnections: Math.floor(5 + Math.random() * 20), // 5-25
        cacheHitRate: 85 + Math.random() * 10, // 85-95%
      });

      setLastUpdate(new Date());

      // Check for degraded models
      modelHealthData.forEach((model) => {
        if (model.status === 'degraded' || model.status === 'down') {
          addToast('error', `Model ${model.name} is ${model.status}`);
        }
        if (model.errorRate > ERROR_RATE_THRESHOLD) {
          addToast('warning', `High error rate for ${model.name}: ${(model.errorRate * 100).toFixed(1)}%`);
        }
      });
    } catch (error) {
      console.error('Error fetching health data:', error);
      trackAPICall('/health', 0, false);
    }
  }, [trackAPICall, addToast]);

  // Calculate predictions per minute
  useEffect(() => {
    const oneMinuteAgo = Date.now() - 60000;
    const recentPredictions = apiMetrics.filter(
      (m) => m.timestamp > oneMinuteAgo && m.endpoint.includes('predict')
    );
    setPredictionsPerMinute(recentPredictions.length);
  }, [apiMetrics]);

  // Load historical data from localStorage
  useEffect(() => {
    const stored = localStorage.getItem('performance_historical_data');
    if (stored) {
      try {
        setHistoricalData(JSON.parse(stored));
      } catch (error) {
        console.error('Error loading historical data:', error);
      }
    } else {
      // Generate mock historical data for last 7 days
      const mockData: HistoricalData[] = [];
      for (let i = 6; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        mockData.push({
          date: date.toISOString().split('T')[0],
          predictions: Math.floor(100 + Math.random() * 400),
          avgAccuracy: 0.85 + Math.random() * 0.1,
          errorRate: Math.random() * 0.05,
        });
      }
      setHistoricalData(mockData);
      localStorage.setItem('performance_historical_data', JSON.stringify(mockData));
    }
  }, []);

  // Polling effect
  useEffect(() => {
    if (!isPolling) return;

    // Initial fetch
    fetchHealthData();

    // Set up polling
    const interval = setInterval(fetchHealthData, POLLING_INTERVAL);

    return () => clearInterval(interval);
  }, [isPolling, fetchHealthData]);

  // Calculate statistics
  const statistics = useMemo(() => {
    const recentMetrics = apiMetrics.slice(-20); // Last 20 requests
    const totalRequests = recentMetrics.length;
    const successfulRequests = recentMetrics.filter((m) => m.success).length;
    const failedRequests = totalRequests - successfulRequests;
    const successRate = totalRequests > 0 ? (successfulRequests / totalRequests) * 100 : 100;
    const avgResponseTime =
      totalRequests > 0
        ? recentMetrics.reduce((sum, m) => sum + m.responseTime, 0) / totalRequests
        : 0;

    return {
      totalRequests,
      successfulRequests,
      failedRequests,
      successRate,
      avgResponseTime,
    };
  }, [apiMetrics]);

  // Response time chart data
  const responseTimeChartData = useMemo(() => {
    const recentMetrics = apiMetrics.slice(-20);
    return {
      labels: recentMetrics.map((_, i) => `${i + 1}`),
      datasets: [
        {
          label: 'Response Time (ms)',
          data: recentMetrics.map((m) => m.responseTime),
          borderColor: 'rgba(59, 130, 246, 1)',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          fill: true,
          tension: 0.4,
        },
        {
          label: 'Threshold',
          data: recentMetrics.map(() => RESPONSE_TIME_THRESHOLD),
          borderColor: 'rgba(239, 68, 68, 0.5)',
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false,
        },
      ],
    };
  }, [apiMetrics]);

  // Success/Error rate chart
  const successRateChartData = useMemo(() => {
    return {
      labels: ['Success', 'Failed'],
      datasets: [
        {
          data: [statistics.successfulRequests, statistics.failedRequests],
          backgroundColor: [
            'rgba(34, 197, 94, 0.8)',
            'rgba(239, 68, 68, 0.8)',
          ],
          borderColor: [
            'rgba(34, 197, 94, 1)',
            'rgba(239, 68, 68, 1)',
          ],
          borderWidth: 2,
        },
      ],
    };
  }, [statistics]);

  // Historical predictions chart
  const historicalPredictionsData = useMemo(() => {
    return {
      labels: historicalData.map((d) => {
        const date = new Date(d.date);
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      }),
      datasets: [
        {
          label: 'Predictions',
          data: historicalData.map((d) => d.predictions),
          backgroundColor: 'rgba(99, 102, 241, 0.8)',
          borderColor: 'rgba(99, 102, 241, 1)',
          borderWidth: 1,
        },
      ],
    };
  }, [historicalData]);

  // Error rate trend chart
  const errorRateTrendData = useMemo(() => {
    return {
      labels: historicalData.map((d) => {
        const date = new Date(d.date);
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      }),
      datasets: [
        {
          label: 'Error Rate (%)',
          data: historicalData.map((d) => d.errorRate * 100),
          borderColor: 'rgba(239, 68, 68, 1)',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          fill: true,
          tension: 0.4,
        },
      ],
    };
  }, [historicalData]);

  // Get status color
  const getStatusColor = (status: ModelHealth['status']) => {
    switch (status) {
      case 'healthy':
        return 'text-success-600 bg-success-100 dark:bg-success-900/30';
      case 'degraded':
        return 'text-warning-600 bg-warning-100 dark:bg-warning-900/30';
      case 'down':
        return 'text-danger-600 bg-danger-100 dark:bg-danger-900/30';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  // Get status icon
  const getStatusIcon = (status: ModelHealth['status']) => {
    switch (status) {
      case 'healthy':
        return <FiCheckCircle className="w-4 h-4" />;
      case 'degraded':
        return <FiAlertCircle className="w-4 h-4" />;
      case 'down':
        return <FiX className="w-4 h-4" />;
    }
  };

  // Get toast icon and color
  const getToastStyle = (type: ToastNotification['type']) => {
    switch (type) {
      case 'success':
        return {
          icon: <FiCheckCircle className="w-5 h-5" />,
          bgClass: 'bg-success-50 dark:bg-success-900/30 border-success-200 dark:border-success-800',
          textClass: 'text-success-900 dark:text-success-100',
          iconClass: 'text-success-600',
        };
      case 'warning':
        return {
          icon: <FiAlertCircle className="w-5 h-5" />,
          bgClass: 'bg-warning-50 dark:bg-warning-900/30 border-warning-200 dark:border-warning-800',
          textClass: 'text-warning-900 dark:text-warning-100',
          iconClass: 'text-warning-600',
        };
      case 'error':
        return {
          icon: <FiAlertCircle className="w-5 h-5" />,
          bgClass: 'bg-danger-50 dark:bg-danger-900/30 border-danger-200 dark:border-danger-800',
          textClass: 'text-danger-900 dark:text-danger-100',
          iconClass: 'text-danger-600',
        };
      case 'info':
        return {
          icon: <FiActivity className="w-5 h-5" />,
          bgClass: 'bg-blue-50 dark:bg-blue-900/30 border-blue-200 dark:border-blue-800',
          textClass: 'text-blue-900 dark:text-blue-100',
          iconClass: 'text-blue-600',
        };
    }
  };

  return (
    <div className="space-y-6">
      {/* Toast Notifications */}
      <div className="fixed top-4 right-4 z-50 space-y-2 max-w-md">
        {toasts.map((toast) => {
          const style = getToastStyle(toast.type);
          return (
            <div
              key={toast.id}
              className={`flex items-start gap-3 p-4 rounded-lg border shadow-lg animate-slide-in ${style.bgClass}`}
            >
              <div className={style.iconClass}>{style.icon}</div>
              <p className={`flex-1 text-sm font-medium ${style.textClass}`}>{toast.message}</p>
              <button
                onClick={() => removeToast(toast.id)}
                className={`${style.iconClass} hover:opacity-70 transition-opacity`}
              >
                <FiX className="w-4 h-4" />
              </button>
            </div>
          );
        })}
      </div>

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
            Performance Monitor
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Real-time system metrics and API performance
          </p>
        </div>

        <div className="flex items-center gap-4">
          {lastUpdate && (
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Last update: {lastUpdate.toLocaleTimeString()}
            </div>
          )}
          <button
            onClick={() => setIsPolling(!isPolling)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              isPolling
                ? 'bg-success-100 text-success-700 dark:bg-success-900/30 dark:text-success-400'
                : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-400'
            }`}
          >
            <div className="flex items-center gap-2">
              <FiRefreshCw className={`w-4 h-4 ${isPolling ? 'animate-spin' : ''}`} />
              {isPolling ? 'Auto-refresh ON' : 'Auto-refresh OFF'}
            </div>
          </button>
        </div>
      </div>

      {/* Real-time Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-blue-600 dark:text-blue-400">
                Avg Response Time
              </p>
              <p className="text-3xl font-bold text-blue-900 dark:text-blue-100 mt-1">
                {statistics.avgResponseTime.toFixed(0)}ms
              </p>
            </div>
            <FiClock className="w-8 h-8 text-blue-600 dark:text-blue-400 opacity-50" />
          </div>
        </Card>

        <Card className="bg-gradient-to-br from-success-50 to-success-100 dark:from-success-900/20 dark:to-success-800/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-success-600 dark:text-success-400">
                Success Rate
              </p>
              <p className="text-3xl font-bold text-success-900 dark:text-success-100 mt-1">
                {statistics.successRate.toFixed(1)}%
              </p>
            </div>
            <FiCheckCircle className="w-8 h-8 text-success-600 dark:text-success-400 opacity-50" />
          </div>
        </Card>

        <Card className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-purple-600 dark:text-purple-400">
                Predictions/Min
              </p>
              <p className="text-3xl font-bold text-purple-900 dark:text-purple-100 mt-1">
                {predictionsPerMinute}
              </p>
            </div>
            <FiZap className="w-8 h-8 text-purple-600 dark:text-purple-400 opacity-50" />
          </div>
        </Card>

        <Card className="bg-gradient-to-br from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-orange-600 dark:text-orange-400">
                Total Requests
              </p>
              <p className="text-3xl font-bold text-orange-900 dark:text-orange-100 mt-1">
                {statistics.totalRequests}
              </p>
            </div>
            <FiActivity className="w-8 h-8 text-orange-600 dark:text-orange-400 opacity-50" />
          </div>
        </Card>
      </div>

      {/* API Performance Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Response Time Trend */}
        <Card title="Response Time Trend" subtitle="Last 20 requests" className="lg:col-span-2">
          <div className="h-64">
            <Line
              data={responseTimeChartData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    position: 'top',
                  },
                },
                scales: {
                  y: {
                    beginAtZero: true,
                    title: {
                      display: true,
                      text: 'Response Time (ms)',
                    },
                  },
                  x: {
                    title: {
                      display: true,
                      text: 'Request Number',
                    },
                  },
                },
              }}
            />
          </div>
        </Card>

        {/* Success/Error Rate */}
        <Card title="Request Status" subtitle="Success vs Failed">
          <div className="h-64">
            <Doughnut
              data={successRateChartData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    position: 'bottom',
                  },
                },
              }}
            />
          </div>
        </Card>
      </div>

      {/* Model Health Indicators */}
      <Card title="Model Health Status" subtitle="Real-time model performance indicators">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {modelHealth.map((model) => (
            <div
              key={model.name}
              className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg"
            >
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-semibold text-gray-900 dark:text-gray-100">
                  {model.name}
                </h4>
                <span
                  className={`px-2 py-1 rounded-full text-xs font-medium flex items-center gap-1 ${getStatusColor(
                    model.status
                  )}`}
                >
                  {getStatusIcon(model.status)}
                  {model.status}
                </span>
              </div>

              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Avg Confidence:</span>
                  <span className="font-medium text-gray-900 dark:text-gray-100">
                    {(model.avgConfidence * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Error Rate:</span>
                  <span
                    className={`font-medium ${
                      model.errorRate > ERROR_RATE_THRESHOLD
                        ? 'text-danger-600'
                        : 'text-success-600'
                    }`}
                  >
                    {(model.errorRate * 100).toFixed(2)}%
                  </span>
                </div>

                {model.lastPrediction && (
                  <div className="flex justify-between">
                    <span className="text-gray-600 dark:text-gray-400">Last Prediction:</span>
                    <span className="font-medium text-gray-900 dark:text-gray-100">
                      {Math.floor((Date.now() - model.lastPrediction) / 1000)}s ago
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* System Metrics */}
      <Card title="System Metrics" subtitle="Resource utilization and performance">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {/* Memory Usage */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <FiDatabase className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Memory
                </span>
              </div>
              <span className="text-sm font-bold text-gray-900 dark:text-gray-100">
                {systemMetrics.memoryUsage.toFixed(1)}%
              </span>
            </div>
            <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all ${
                  systemMetrics.memoryUsage > 80
                    ? 'bg-danger-600'
                    : systemMetrics.memoryUsage > 60
                    ? 'bg-warning-600'
                    : 'bg-success-600'
                }`}
                style={{ width: `${systemMetrics.memoryUsage}%` }}
              />
            </div>
          </div>

          {/* CPU Usage */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <FiCpu className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  CPU
                </span>
              </div>
              <span className="text-sm font-bold text-gray-900 dark:text-gray-100">
                {systemMetrics.cpuUsage.toFixed(1)}%
              </span>
            </div>
            <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all ${
                  systemMetrics.cpuUsage > 80
                    ? 'bg-danger-600'
                    : systemMetrics.cpuUsage > 60
                    ? 'bg-warning-600'
                    : 'bg-success-600'
                }`}
                style={{ width: `${systemMetrics.cpuUsage}%` }}
              />
            </div>
          </div>

          {/* Active Connections */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <FiServer className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Connections
                </span>
              </div>
              <span className="text-sm font-bold text-gray-900 dark:text-gray-100">
                {systemMetrics.activeConnections}
              </span>
            </div>
            <div className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
              {systemMetrics.activeConnections > 20 ? (
                <>
                  <FiTrendingUp className="w-3 h-3 text-danger-600" />
                  <span>High load</span>
                </>
              ) : (
                <>
                  <FiTrendingDown className="w-3 h-3 text-success-600" />
                  <span>Normal</span>
                </>
              )}
            </div>
          </div>

          {/* Cache Hit Rate */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <FiZap className="w-4 h-4 text-gray-600 dark:text-gray-400" />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Cache Hit Rate
                </span>
              </div>
              <span className="text-sm font-bold text-gray-900 dark:text-gray-100">
                {systemMetrics.cacheHitRate.toFixed(1)}%
              </span>
            </div>
            <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary-600 transition-all"
                style={{ width: `${systemMetrics.cacheHitRate}%` }}
              />
            </div>
          </div>
        </div>
      </Card>

      {/* Historical Performance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Daily Predictions */}
        <Card title="Daily Prediction Volume" subtitle="Last 7 days">
          <div className="h-64">
            <Bar
              data={historicalPredictionsData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    display: false,
                  },
                },
                scales: {
                  y: {
                    beginAtZero: true,
                    title: {
                      display: true,
                      text: 'Number of Predictions',
                    },
                  },
                },
              }}
            />
          </div>
        </Card>

        {/* Error Rate Trend */}
        <Card title="Error Rate Trend" subtitle="Last 7 days">
          <div className="h-64">
            <Line
              data={errorRateTrendData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: {
                    display: false,
                  },
                },
                scales: {
                  y: {
                    beginAtZero: true,
                    title: {
                      display: true,
                      text: 'Error Rate (%)',
                    },
                  },
                },
              }}
            />
          </div>
        </Card>
      </div>

      {/* Recent API Calls */}
      <Card title="Recent API Calls" subtitle="Last 10 requests">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                <th className="px-3 py-3 text-left font-medium text-gray-700 dark:text-gray-300">
                  Time
                </th>
                <th className="px-3 py-3 text-left font-medium text-gray-700 dark:text-gray-300">
                  Endpoint
                </th>
                <th className="px-3 py-3 text-left font-medium text-gray-700 dark:text-gray-300">
                  Response Time
                </th>
                <th className="px-3 py-3 text-left font-medium text-gray-700 dark:text-gray-300">
                  Status
                </th>
              </tr>
            </thead>
            <tbody>
              {apiMetrics.slice(-10).reverse().map((metric, index) => (
                <tr
                  key={`${metric.timestamp}-${index}`}
                  className="border-t border-gray-200 dark:border-gray-700"
                >
                  <td className="px-3 py-3 text-gray-600 dark:text-gray-400">
                    {new Date(metric.timestamp).toLocaleTimeString()}
                  </td>
                  <td className="px-3 py-3 font-mono text-xs text-gray-900 dark:text-gray-100">
                    {metric.endpoint}
                  </td>
                  <td className="px-3 py-3">
                    <span
                      className={`font-medium ${
                        metric.responseTime > RESPONSE_TIME_THRESHOLD
                          ? 'text-danger-600'
                          : metric.responseTime > RESPONSE_TIME_THRESHOLD / 2
                          ? 'text-warning-600'
                          : 'text-success-600'
                      }`}
                    >
                      {metric.responseTime}ms
                    </span>
                  </td>
                  <td className="px-3 py-3">
                    {metric.success ? (
                      <span className="px-2 py-1 rounded-full text-xs font-medium text-success-600 bg-success-100 dark:bg-success-900/30">
                        Success
                      </span>
                    ) : (
                      <span className="px-2 py-1 rounded-full text-xs font-medium text-danger-600 bg-danger-100 dark:bg-danger-900/30">
                        Failed
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {apiMetrics.length === 0 && (
            <div className="text-center py-12">
              <p className="text-gray-500 dark:text-gray-400">
                No API calls tracked yet. Metrics will appear as you use the application.
              </p>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};

export default PerformanceMonitor;
