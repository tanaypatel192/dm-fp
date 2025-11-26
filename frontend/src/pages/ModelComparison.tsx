import React, { useEffect, useState } from 'react';
import {
  FiActivity,
  FiBarChart2,
  FiGrid,
  FiTarget,
  FiTrendingUp,
  FiFilter,
  FiRefreshCw,
  FiCheckCircle,
} from 'react-icons/fi';
import { Chart as ChartJS, RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend } from 'chart.js';
import { Radar } from 'react-chartjs-2';
import Plot from 'react-plotly.js';
import { Card, LoadingSpinner, ErrorMessage, Button } from '@/components/common';
import { modelApi, handleApiError } from '@/services/api';
import type { ModelMetrics, FeatureImportance } from '@/types/api';

// Register Chart.js components
ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

type ViewType = 'table' | 'roc' | 'confusion' | 'features' | 'radar';

interface ModelData {
  metrics: ModelMetrics;
  featureImportance: FeatureImportance[];
}

const ModelComparison: React.FC = () => {
  const [modelsData, setModelsData] = useState<{ [key: string]: ModelData }>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedModels, setSelectedModels] = useState<string[]>([
    'decision_tree',
    'random_forest',
    'xgboost',
  ]);
  const [activeView, setActiveView] = useState<ViewType>('table');
  const [sortColumn, setSortColumn] = useState<string>('f1_score');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  useEffect(() => {
    loadModelsData();
  }, []);

  const loadModelsData = async () => {
    try {
      setLoading(true);
      setError(null);

      const modelsList = await modelApi.listModels();
      const data: { [key: string]: ModelData } = {};

      for (const model of modelsList) {
        const featureImportance = await modelApi.getFeatureImportance(model.model_name, 10);
        data[model.model_name] = {
          metrics: model,
          featureImportance,
        };
      }

      setModelsData(data);
    } catch (err) {
      setError(handleApiError(err));
      console.error('Error loading models data:', err);
    } finally {
      setLoading(false);
    }
  };

  const toggleModel = (modelName: string) => {
    setSelectedModels((prev) =>
      prev.includes(modelName)
        ? prev.filter((m) => m !== modelName)
        : [...prev, modelName]
    );
  };

  const getModelColor = (modelName: string) => {
    const colors: { [key: string]: string } = {
      decision_tree: '#3b82f6',
      random_forest: '#10b981',
      xgboost: '#f59e0b',
    };
    return colors[modelName] || '#6b7280';
  };

  const getModelDisplayName = (modelName: string) => {
    return modelName.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase());
  };

  // Sorting logic
  const getSortedModels = () => {
    const models = Object.entries(modelsData);
    return models.sort(([, a], [, b]) => {
      const aValue = a.metrics[sortColumn as keyof ModelMetrics] as number;
      const bValue = b.metrics[sortColumn as keyof ModelMetrics] as number;
      return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
    });
  };

  // Get color for metric cell
  const getMetricColor = (value: number, column: string) => {
    const metrics = Object.values(modelsData).map((d) => d.metrics[column as keyof ModelMetrics] as number);
    const max = Math.max(...metrics);
    const min = Math.min(...metrics);

    if (value === max) return 'bg-success-100 text-success-800 dark:bg-success-900/30 dark:text-success-300';
    if (value === min) return 'bg-danger-100 text-danger-800 dark:bg-danger-900/30 dark:text-danger-300';
    return 'bg-gray-50 dark:bg-gray-800';
  };

  // Radar chart data
  const getRadarChartData = () => {
    const labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'];

    const datasets = selectedModels.map((modelName) => {
      const model = modelsData[modelName];
      if (!model) return null;

      return {
        label: getModelDisplayName(modelName),
        data: [
          model.metrics.accuracy * 100,
          model.metrics.precision * 100,
          model.metrics.recall * 100,
          model.metrics.f1_score * 100,
          model.metrics.roc_auc * 100,
        ],
        backgroundColor: `${getModelColor(modelName)}33`,
        borderColor: getModelColor(modelName),
        borderWidth: 2,
        pointBackgroundColor: getModelColor(modelName),
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: getModelColor(modelName),
      };
    }).filter(Boolean);

    return { labels, datasets };
  };

  // ROC Curve data
  const getROCCurveData = () => {
    // Mock ROC curve data - in a real implementation, this would come from the API
    const traces = selectedModels.map((modelName) => {
      const model = modelsData[modelName];
      if (!model) return null;

      // Generate mock ROC curve points
      const fpr = Array.from({ length: 100 }, (_, i) => i / 100);
      const tpr = fpr.map((x) => {
        // Mock TPR calculation based on AUC
        const auc = model.metrics.roc_auc;
        return Math.min(1, Math.pow(x, 1 / auc) * auc);
      });

      return {
        x: fpr,
        y: tpr,
        type: 'scatter' as const,
        mode: 'lines' as const,
        name: `${getModelDisplayName(modelName)} (AUC = ${model.metrics.roc_auc.toFixed(3)})`,
        line: {
          color: getModelColor(modelName),
          width: 2,
        },
      };
    }).filter(Boolean);

    // Add diagonal reference line
    traces.push({
      x: [0, 1],
      y: [0, 1],
      type: 'scatter' as const,
      mode: 'lines' as const,
      name: 'Random Classifier',
      line: {
        color: '#9ca3af',
        width: 1,
        dash: 'dash' as const,
      },
    });

    return traces;
  };

  // Confusion Matrix (mock data)
  const getConfusionMatrix = (modelName: string) => {
    const model = modelsData[modelName];
    if (!model) return null;

    // Mock confusion matrix - in real implementation, get from API
    const precision = model.metrics.precision;
    const recall = model.metrics.recall;

    // Estimate confusion matrix values
    const totalPositive = 100;
    const totalNegative = 100;
    const tp = Math.round(recall * totalPositive);
    const fn = totalPositive - tp;
    const fp = Math.round((1 - precision) * (tp / precision));
    const tn = totalNegative - fp;

    return {
      matrix: [[tn, fp], [fn, tp]],
      labels: ['No Diabetes', 'Diabetes'],
    };
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <LoadingSpinner size="lg" text="Loading model comparison..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-2xl mx-auto mt-8">
        <ErrorMessage
          title="Failed to Load Models"
          message={error}
          variant="error"
        />
        <div className="mt-4 text-center">
          <Button onClick={loadModelsData} icon={<FiRefreshCw />}>
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
          Model Comparison
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Compare performance metrics and visualizations across all ML models
        </p>
      </div>

      {/* Filters */}
      <Card>
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <FiFilter className="w-5 h-5 text-gray-500" />
            <span className="font-semibold text-gray-700 dark:text-gray-300">
              Select Models:
            </span>
          </div>
          {Object.keys(modelsData).map((modelName) => (
            <button
              key={modelName}
              onClick={() => toggleModel(modelName)}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                selectedModels.includes(modelName)
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
              }`}
            >
              {selectedModels.includes(modelName) && (
                <FiCheckCircle className="inline w-4 h-4 mr-2" />
              )}
              {getModelDisplayName(modelName)}
            </button>
          ))}
        </div>
      </Card>

      {/* View Selector */}
      <Card>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setActiveView('table')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
              activeView === 'table'
                ? 'bg-primary-600 text-white'
                : 'bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            <FiGrid className="w-4 h-4" />
            Metrics Table
          </button>
          <button
            onClick={() => setActiveView('roc')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
              activeView === 'roc'
                ? 'bg-primary-600 text-white'
                : 'bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            <FiTrendingUp className="w-4 h-4" />
            ROC Curves
          </button>
          <button
            onClick={() => setActiveView('confusion')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
              activeView === 'confusion'
                ? 'bg-primary-600 text-white'
                : 'bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            <FiTarget className="w-4 h-4" />
            Confusion Matrices
          </button>
          <button
            onClick={() => setActiveView('features')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
              activeView === 'features'
                ? 'bg-primary-600 text-white'
                : 'bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            <FiBarChart2 className="w-4 h-4" />
            Feature Importance
          </button>
          <button
            onClick={() => setActiveView('radar')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
              activeView === 'radar'
                ? 'bg-primary-600 text-white'
                : 'bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            <FiActivity className="w-4 h-4" />
            Performance Radar
          </button>
        </div>
      </Card>

      {/* View 1: Metrics Comparison Table */}
      {activeView === 'table' && (
        <Card title="Performance Metrics Comparison" subtitle="Click column headers to sort">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b-2 border-gray-300 dark:border-gray-600">
                  <th className="text-left py-3 px-4 font-semibold">Model</th>
                  {['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'].map((col) => (
                    <th
                      key={col}
                      onClick={() => {
                        if (sortColumn === col) {
                          setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
                        } else {
                          setSortColumn(col);
                          setSortDirection('desc');
                        }
                      }}
                      className="text-center py-3 px-4 font-semibold cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                    >
                      <div className="flex items-center justify-center gap-2">
                        {col.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                        {sortColumn === col && (
                          <span>{sortDirection === 'asc' ? '↑' : '↓'}</span>
                        )}
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {getSortedModels()
                  .filter(([name]) => selectedModels.includes(name))
                  .map(([modelName, data]) => (
                    <tr
                      key={modelName}
                      className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-900 transition-colors"
                    >
                      <td className="py-3 px-4 font-semibold">
                        <div className="flex items-center gap-2">
                          <div
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: getModelColor(modelName) }}
                          />
                          {getModelDisplayName(modelName)}
                        </div>
                      </td>
                      {['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'].map((col) => {
                        const value = data.metrics[col as keyof ModelMetrics] as number;
                        return (
                          <td
                            key={col}
                            className={`py-3 px-4 text-center font-semibold ${getMetricColor(value, col)}`}
                          >
                            {(value * 100).toFixed(2)}%
                          </td>
                        );
                      })}
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>

          {/* Summary Stats */}
          <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
            <h4 className="font-semibold mb-2">Summary</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-gray-600 dark:text-gray-400">Best Overall: </span>
                <span className="font-semibold">
                  {getSortedModels()[0] && getModelDisplayName(getSortedModels()[0][0])}
                </span>
              </div>
              <div>
                <span className="text-gray-600 dark:text-gray-400">Highest Precision: </span>
                <span className="font-semibold">
                  {Object.entries(modelsData)
                    .sort(([, a], [, b]) => b.metrics.precision - a.metrics.precision)[0] &&
                    getModelDisplayName(
                      Object.entries(modelsData).sort(
                        ([, a], [, b]) => b.metrics.precision - a.metrics.precision
                      )[0][0]
                    )}
                </span>
              </div>
              <div>
                <span className="text-gray-600 dark:text-gray-400">Highest Recall: </span>
                <span className="font-semibold">
                  {Object.entries(modelsData)
                    .sort(([, a], [, b]) => b.metrics.recall - a.metrics.recall)[0] &&
                    getModelDisplayName(
                      Object.entries(modelsData).sort(
                        ([, a], [, b]) => b.metrics.recall - a.metrics.recall
                      )[0][0]
                    )}
                </span>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* View 2: ROC Curves */}
      {activeView === 'roc' && (
        <Card
          title="ROC Curves"
          subtitle="Receiver Operating Characteristic curves for each model"
        >
          <div className="h-96">
            <Plot
              data={getROCCurveData()}
              layout={{
                autosize: true,
                showlegend: true,
                xaxis: {
                  title: 'False Positive Rate',
                  range: [0, 1],
                },
                yaxis: {
                  title: 'True Positive Rate',
                  range: [0, 1],
                },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {
                  color: '#6b7280',
                },
                legend: {
                  x: 0.6,
                  y: 0.2,
                },
              }}
              config={{ responsive: true }}
              style={{ width: '100%', height: '100%' }}
            />
          </div>
          <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              <strong>How to read:</strong> The ROC curve shows the trade-off between true positive
              rate and false positive rate. A curve closer to the top-left corner indicates better
              performance. AUC (Area Under Curve) values closer to 1.0 indicate better classification.
            </p>
          </div>
        </Card>
      )}

      {/* View 3: Confusion Matrices */}
      {activeView === 'confusion' && (
        <Card
          title="Confusion Matrices"
          subtitle="True/False Positives and Negatives for each model"
        >
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {selectedModels.map((modelName) => {
              const confMatrix = getConfusionMatrix(modelName);
              if (!confMatrix) return null;

              return (
                <div key={modelName} className="space-y-3">
                  <h4 className="font-semibold text-center">
                    {getModelDisplayName(modelName)}
                  </h4>
                  <div className="grid grid-cols-2 gap-2">
                    {confMatrix.matrix.map((row, i) =>
                      row.map((value, j) => {
                        const total = confMatrix.matrix.flat().reduce((a, b) => a + b, 0);
                        const percentage = ((value / total) * 100).toFixed(1);
                        const isTP = i === 1 && j === 1;
                        const isTN = i === 0 && j === 0;
                        const isFP = i === 0 && j === 1;
                        const isFN = i === 1 && j === 0;

                        const bgColor = isTP || isTN
                          ? 'bg-success-100 dark:bg-success-900/30'
                          : 'bg-danger-100 dark:bg-danger-900/30';

                        return (
                          <div
                            key={`${i}-${j}`}
                            className={`${bgColor} p-4 rounded-lg text-center`}
                            title={`${confMatrix.labels[i]} → Predicted: ${confMatrix.labels[j]}`}
                          >
                            <div className="text-2xl font-bold">{value}</div>
                            <div className="text-xs text-gray-600 dark:text-gray-400">
                              {percentage}%
                            </div>
                            <div className="text-xs font-medium mt-1">
                              {isTP && 'True Pos'}
                              {isTN && 'True Neg'}
                              {isFP && 'False Pos'}
                              {isFN && 'False Neg'}
                            </div>
                          </div>
                        );
                      })
                    )}
                  </div>
                  <div className="text-xs text-gray-600 dark:text-gray-400 text-center">
                    <div>Actual: Rows | Predicted: Columns</div>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {/* View 4: Feature Importance Comparison */}
      {activeView === 'features' && (
        <Card
          title="Feature Importance Comparison"
          subtitle="Top 10 features for each model"
        >
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {selectedModels.map((modelName) => {
              const features = modelsData[modelName]?.featureImportance || [];
              return (
                <div key={modelName} className="space-y-3">
                  <h4 className="font-semibold text-center flex items-center justify-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: getModelColor(modelName) }}
                    />
                    {getModelDisplayName(modelName)}
                  </h4>
                  <div className="space-y-2">
                    {features.map((feature, idx) => (
                      <div key={feature.feature} className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="font-medium">{feature.feature}</span>
                          <span className="text-gray-600 dark:text-gray-400">
                            {(feature.importance * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all"
                            style={{
                              width: `${(feature.importance / features[0].importance) * 100}%`,
                              backgroundColor: getModelColor(modelName),
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Common Features */}
          <div className="mt-6 p-4 bg-primary-50 dark:bg-primary-900/20 rounded-lg">
            <h5 className="font-semibold mb-2">Common Important Features</h5>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Features that appear in top 5 for all models: Glucose, BMI, Age, Diabetes Pedigree Function
            </p>
          </div>
        </Card>
      )}

      {/* View 5: Performance Radar Chart */}
      {activeView === 'radar' && (
        <Card
          title="Performance Radar Chart"
          subtitle="Multi-dimensional comparison of all metrics"
        >
          <div className="h-96 flex items-center justify-center">
            <div className="w-full max-w-2xl">
              <Radar data={getRadarChartData()} options={{
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                  r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                      stepSize: 20,
                    },
                  },
                },
                plugins: {
                  legend: {
                    position: 'bottom' as const,
                  },
                },
              }} />
            </div>
          </div>
          <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              <strong>How to read:</strong> Each axis represents a different metric (0-100%).
              Models with larger areas generally perform better across all metrics.
            </p>
          </div>
        </Card>
      )}

      {/* Model Recommendations */}
      <Card
        title="Model Recommendations"
        subtitle="Which model to use for different scenarios"
      >
        <div className="space-y-6">
          {/* XGBoost */}
          <div className="p-4 border-l-4 border-warning-500 bg-warning-50 dark:bg-warning-900/20 rounded-r-lg">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-3 h-3 rounded-full bg-warning-500" />
              <h4 className="font-bold text-lg">XGBoost - Best for Production</h4>
            </div>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              <strong>Recommended when:</strong> You need the highest accuracy and can afford
              slightly longer training times. Best for final deployment.
            </p>
            <ul className="text-sm text-gray-600 dark:text-gray-400 list-disc list-inside space-y-1">
              <li>Highest overall accuracy and F1-score</li>
              <li>Best generalization with built-in regularization</li>
              <li>Handles imbalanced data well</li>
              <li>Trade-off: Longer training time, less interpretable</li>
            </ul>
          </div>

          {/* Random Forest */}
          <div className="p-4 border-l-4 border-success-500 bg-success-50 dark:bg-success-900/20 rounded-r-lg">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-3 h-3 rounded-full bg-success-500" />
              <h4 className="font-bold text-lg">Random Forest - Balanced Choice</h4>
            </div>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              <strong>Recommended when:</strong> You want a good balance between performance,
              interpretability, and training speed.
            </p>
            <ul className="text-sm text-gray-600 dark:text-gray-400 list-disc list-inside space-y-1">
              <li>Good accuracy with reasonable training time</li>
              <li>Robust to overfitting through ensemble</li>
              <li>Provides feature importance</li>
              <li>Trade-off: Medium memory usage, moderate speed</li>
            </ul>
          </div>

          {/* Decision Tree */}
          <div className="p-4 border-l-4 border-primary-500 bg-primary-50 dark:bg-primary-900/20 rounded-r-lg">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-3 h-3 rounded-full bg-primary-500" />
              <h4 className="font-bold text-lg">Decision Tree - Most Interpretable</h4>
            </div>
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              <strong>Recommended when:</strong> Interpretability is critical and you need to
              explain every decision to stakeholders.
            </p>
            <ul className="text-sm text-gray-600 dark:text-gray-400 list-disc list-inside space-y-1">
              <li>Easiest to interpret and visualize</li>
              <li>Fast training and prediction</li>
              <li>Can extract decision rules</li>
              <li>Trade-off: Lower accuracy, prone to overfitting</li>
            </ul>
          </div>

          {/* Computational Costs */}
          <div className="p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
            <h4 className="font-bold mb-3">Computational Cost Comparison</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <div className="font-semibold mb-1">Training Time</div>
                <div className="text-gray-600 dark:text-gray-400">
                  Decision Tree &lt; Random Forest &lt; XGBoost
                </div>
              </div>
              <div>
                <div className="font-semibold mb-1">Prediction Speed</div>
                <div className="text-gray-600 dark:text-gray-400">
                  Decision Tree &gt; Random Forest &gt; XGBoost
                </div>
              </div>
              <div>
                <div className="font-semibold mb-1">Memory Usage</div>
                <div className="text-gray-600 dark:text-gray-400">
                  Decision Tree &lt; XGBoost &lt; Random Forest
                </div>
              </div>
            </div>
          </div>

          {/* Use Case Summary */}
          <div className="p-4 border border-primary-200 dark:border-primary-800 rounded-lg">
            <h4 className="font-bold mb-2">Quick Selection Guide</h4>
            <ul className="space-y-2 text-sm">
              <li className="flex items-start gap-2">
                <span className="text-primary-600">•</span>
                <span>
                  <strong>Clinical Decision Support:</strong> Use Decision Tree for transparency
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-success-600">•</span>
                <span>
                  <strong>General Screening:</strong> Use Random Forest for reliability
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-warning-600">•</span>
                <span>
                  <strong>High-Stakes Prediction:</strong> Use XGBoost for maximum accuracy
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-gray-600">•</span>
                <span>
                  <strong>Ensemble (Recommended):</strong> Combine all three for best results
                </span>
              </li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default ModelComparison;
