/**
 * A/B Test Analytics Dashboard
 *
 * Comprehensive dashboard for viewing A/B test results and metrics
 */

import React, { useState } from 'react';
import {
  FiTrendingUp,
  FiTrendingDown,
  FiMinus,
  FiCheckCircle,
  FiXCircle,
  FiAlertCircle,
  FiUsers,
  FiActivity,
  FiClock,
  FiStar,
  FiBarChart2,
} from 'react-icons/fi';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import {
  useExperiment,
  useExperimentResults,
  type Experiment,
  type ExperimentResults,
  type VariantMetrics,
  type StatisticalComparison,
} from '@/hooks/useABTest';

interface ABTestDashboardProps {
  experimentId: string;
}

const ABTestDashboard: React.FC<ABTestDashboardProps> = ({ experimentId }) => {
  const { data: experiment, isLoading: isLoadingExperiment } = useExperiment(experimentId);
  const { data: results, isLoading: isLoadingResults } = useExperimentResults(experimentId);

  const [selectedVariant, setSelectedVariant] = useState<string | null>(null);

  if (isLoadingExperiment || isLoadingResults) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading experiment results...</p>
        </div>
      </div>
    );
  }

  if (!experiment || !results) {
    return (
      <div className="text-center py-12">
        <FiAlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <p className="text-gray-600">Experiment not found</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <ExperimentHeader experiment={experiment} results={results} />

      {/* Winner Announcement */}
      {results.winner.winner && (
        <WinnerAnnouncement results={results} experiment={experiment} />
      )}

      {/* Statistical Comparisons */}
      {results.comparisons.length > 0 && (
        <StatisticalComparisons comparisons={results.comparisons} />
      )}

      {/* Variant Metrics Cards */}
      <VariantMetricsGrid experiment={experiment} />

      {/* Detailed Charts */}
      <MetricsCharts experiment={experiment} />

      {/* Variant Details */}
      <VariantDetails
        experiment={experiment}
        selectedVariant={selectedVariant}
        onSelectVariant={setSelectedVariant}
      />
    </div>
  );
};

// ==================== Sub-Components ====================

const ExperimentHeader: React.FC<{
  experiment: Experiment;
  results: ExperimentResults;
}> = ({ experiment, results }) => {
  const getStatusBadge = (status: string) => {
    const colors = {
      draft: 'bg-gray-100 text-gray-800',
      running: 'bg-green-100 text-green-800',
      paused: 'bg-yellow-100 text-yellow-800',
      completed: 'bg-blue-100 text-blue-800',
      cancelled: 'bg-red-100 text-red-800',
    };

    return (
      <span className={`px-3 py-1 rounded-full text-sm font-medium ${colors[status as keyof typeof colors] || colors.draft}`}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    );
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-6">
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <h1 className="text-2xl font-bold text-gray-900">{experiment.name}</h1>
            {getStatusBadge(experiment.status)}
          </div>
          <p className="text-gray-600 mb-4">{experiment.description}</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Created:</span>
              <span className="ml-2 font-medium">{new Date(experiment.created_at).toLocaleDateString()}</span>
            </div>
            {experiment.started_at && (
              <div>
                <span className="text-gray-500">Started:</span>
                <span className="ml-2 font-medium">{new Date(experiment.started_at).toLocaleDateString()}</span>
              </div>
            )}
            <div>
              <span className="text-gray-500">Duration:</span>
              <span className="ml-2 font-medium">{results.duration_hours.toFixed(1)}h</span>
            </div>
            <div>
              <span className="text-gray-500">Target Metric:</span>
              <span className="ml-2 font-medium">{experiment.target_metric.replace('_', ' ')}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const WinnerAnnouncement: React.FC<{
  results: ExperimentResults;
  experiment: Experiment;
}> = ({ results, experiment }) => {
  const winner = results.winner;

  if (!winner.winner) {
    return null;
  }

  const variant = experiment.variants.find((v) => v.id === winner.winner);

  return (
    <div className="bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-200 rounded-lg p-6">
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0">
          <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center">
            <FiCheckCircle className="w-7 h-7 text-white" />
          </div>
        </div>
        <div className="flex-1">
          <h2 className="text-xl font-bold text-gray-900 mb-2">Winner Identified! 🎉</h2>
          <p className="text-gray-700 mb-3">
            <span className="font-semibold">{variant?.name}</span> (using model{' '}
            <span className="font-mono text-sm bg-white px-2 py-1 rounded">{variant?.model_name}</span>)
            has shown a <span className="font-bold text-green-600">+{winner.lift?.toFixed(2)}%</span> improvement
            with {((winner.confidence || 0) * 100).toFixed(0)}% confidence.
          </p>
          <button className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-lg font-medium transition-colors">
            Promote to Production
          </button>
        </div>
      </div>
    </div>
  );
};

const StatisticalComparisons: React.FC<{
  comparisons: StatisticalComparison[];
}> = ({ comparisons }) => {
  return (
    <div className="bg-white rounded-lg shadow-sm p-6">
      <h2 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
        <FiBarChart2 className="w-5 h-5" />
        Statistical Comparisons
      </h2>
      <div className="space-y-4">
        {comparisons.map((comparison, index) => (
          <ComparisonCard key={index} comparison={comparison} />
        ))}
      </div>
    </div>
  );
};

const ComparisonCard: React.FC<{ comparison: StatisticalComparison }> = ({ comparison }) => {
  const getTrendIcon = () => {
    if (comparison.relative_lift_percent > 0) {
      return <FiTrendingUp className="w-5 h-5 text-green-600" />;
    } else if (comparison.relative_lift_percent < 0) {
      return <FiTrendingDown className="w-5 h-5 text-red-600" />;
    }
    return <FiMinus className="w-5 h-5 text-gray-400" />;
  };

  const getSignificanceIcon = () => {
    if (comparison.significant) {
      return <FiCheckCircle className="w-5 h-5 text-green-600" />;
    }
    return <FiXCircle className="w-5 h-5 text-gray-400" />;
  };

  return (
    <div className="border border-gray-200 rounded-lg p-4">
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="font-semibold text-gray-900">{comparison.variant_name}</h3>
          <p className="text-sm text-gray-600">vs Control</p>
        </div>
        <div className="flex items-center gap-3">
          {getTrendIcon()}
          {getSignificanceIcon()}
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
        <div>
          <p className="text-xs text-gray-500">Relative Lift</p>
          <p className={`text-lg font-bold ${comparison.relative_lift_percent > 0 ? 'text-green-600' : comparison.relative_lift_percent < 0 ? 'text-red-600' : 'text-gray-600'}`}>
            {comparison.relative_lift_percent > 0 ? '+' : ''}
            {comparison.relative_lift_percent.toFixed(2)}%
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500">P-Value</p>
          <p className="text-lg font-bold text-gray-900">{comparison.p_value.toFixed(4)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Treatment Mean</p>
          <p className="text-lg font-bold text-gray-900">{comparison.treatment_mean.toFixed(3)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Sample Size</p>
          <p className="text-lg font-bold text-gray-900">{comparison.treatment_sample_size}</p>
        </div>
      </div>

      <div className="bg-gray-50 rounded p-3">
        <p className="text-sm text-gray-700">
          <span className="font-medium">Recommendation:</span> {comparison.recommendation}
        </p>
      </div>
    </div>
  );
};

const VariantMetricsGrid: React.FC<{ experiment: Experiment }> = ({ experiment }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {experiment.variants.map((variant) => {
        const metrics = experiment.metrics[variant.id];
        if (!metrics) return null;

        return (
          <div key={variant.id} className="bg-white rounded-lg shadow-sm p-6">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="font-semibold text-gray-900">{variant.name}</h3>
                <p className="text-sm text-gray-600">{variant.model_name}</p>
              </div>
              <span className={`px-2 py-1 rounded text-xs font-medium ${variant.variant_type === 'control' ? 'bg-blue-100 text-blue-800' : 'bg-purple-100 text-purple-800'}`}>
                {variant.variant_type}
              </span>
            </div>

            <div className="space-y-3">
              <MetricRow
                icon={<FiUsers />}
                label="Users"
                value={metrics.total_users.toString()}
              />
              <MetricRow
                icon={<FiActivity />}
                label="Predictions"
                value={metrics.total_predictions.toString()}
              />
              <MetricRow
                icon={<FiClock />}
                label="Avg Time"
                value={`${metrics.avg_prediction_time_ms.toFixed(0)}ms`}
              />
              <MetricRow
                icon={<FiTrendingUp />}
                label="Conversion"
                value={`${metrics.conversion_rate.toFixed(1)}%`}
              />
              {metrics.rating_count > 0 && (
                <MetricRow
                  icon={<FiStar />}
                  label="Rating"
                  value={`${metrics.avg_rating.toFixed(2)}/5`}
                />
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};

const MetricRow: React.FC<{
  icon: React.ReactNode;
  label: string;
  value: string;
}> = ({ icon, label, value }) => (
  <div className="flex items-center justify-between">
    <div className="flex items-center gap-2 text-gray-600">
      {icon}
      <span className="text-sm">{label}</span>
    </div>
    <span className="font-semibold text-gray-900">{value}</span>
  </div>
);

const MetricsCharts: React.FC<{ experiment: Experiment }> = ({ experiment }) => {
  // Prepare data for conversion rate chart
  const conversionData = {
    labels: experiment.variants.map((v) => v.name),
    datasets: [
      {
        label: 'Conversion Rate (%)',
        data: experiment.variants.map((v) => experiment.metrics[v.id]?.conversion_rate || 0),
        backgroundColor: 'rgba(99, 102, 241, 0.5)',
        borderColor: 'rgba(99, 102, 241, 1)',
        borderWidth: 2,
      },
    ],
  };

  // Prepare data for prediction distribution
  const predictionDistribution = {
    labels: experiment.variants.map((v) => v.name),
    datasets: [
      {
        label: 'Positive Predictions',
        data: experiment.variants.map((v) => experiment.metrics[v.id]?.positive_predictions || 0),
        backgroundColor: 'rgba(34, 197, 94, 0.5)',
      },
      {
        label: 'Negative Predictions',
        data: experiment.variants.map((v) => experiment.metrics[v.id]?.negative_predictions || 0),
        backgroundColor: 'rgba(239, 68, 68, 0.5)',
      },
    ],
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="bg-white rounded-lg shadow-sm p-6">
        <h3 className="font-semibold text-gray-900 mb-4">Conversion Rate by Variant</h3>
        <Bar
          data={conversionData}
          options={{
            responsive: true,
            plugins: {
              legend: { display: false },
            },
            scales: {
              y: {
                beginAtZero: true,
                title: { display: true, text: 'Conversion Rate (%)' },
              },
            },
          }}
        />
      </div>

      <div className="bg-white rounded-lg shadow-sm p-6">
        <h3 className="font-semibold text-gray-900 mb-4">Prediction Distribution</h3>
        <Bar
          data={predictionDistribution}
          options={{
            responsive: true,
            plugins: {
              legend: { position: 'top' },
            },
            scales: {
              y: {
                beginAtZero: true,
                title: { display: true, text: 'Count' },
              },
            },
          }}
        />
      </div>
    </div>
  );
};

const VariantDetails: React.FC<{
  experiment: Experiment;
  selectedVariant: string | null;
  onSelectVariant: (id: string | null) => void;
}> = ({ experiment, selectedVariant, onSelectVariant }) => {
  const variant = experiment.variants.find((v) => v.id === selectedVariant);
  const metrics = variant ? experiment.metrics[variant.id] : null;

  return (
    <div className="bg-white rounded-lg shadow-sm p-6">
      <h2 className="text-lg font-bold text-gray-900 mb-4">Variant Details</h2>

      <div className="flex gap-2 mb-6">
        {experiment.variants.map((v) => (
          <button
            key={v.id}
            onClick={() => onSelectVariant(v.id === selectedVariant ? null : v.id)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedVariant === v.id
                ? 'bg-primary-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            {v.name}
          </button>
        ))}
      </div>

      {variant && metrics && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="border border-gray-200 rounded-lg p-4">
              <p className="text-sm text-gray-600 mb-1">Total Users</p>
              <p className="text-2xl font-bold text-gray-900">{metrics.total_users}</p>
            </div>
            <div className="border border-gray-200 rounded-lg p-4">
              <p className="text-sm text-gray-600 mb-1">Total Predictions</p>
              <p className="text-2xl font-bold text-gray-900">{metrics.total_predictions}</p>
            </div>
            <div className="border border-gray-200 rounded-lg p-4">
              <p className="text-sm text-gray-600 mb-1">Avg Confidence</p>
              <p className="text-2xl font-bold text-gray-900">{metrics.avg_confidence.toFixed(2)}</p>
            </div>
            <div className="border border-gray-200 rounded-lg p-4">
              <p className="text-sm text-gray-600 mb-1">Error Rate</p>
              <p className="text-2xl font-bold text-gray-900">{(metrics.error_rate * 100).toFixed(2)}%</p>
            </div>
            <div className="border border-gray-200 rounded-lg p-4">
              <p className="text-sm text-gray-600 mb-1">Low Risk</p>
              <p className="text-2xl font-bold text-gray-900">{metrics.low_risk_count}</p>
            </div>
            <div className="border border-gray-200 rounded-lg p-4">
              <p className="text-sm text-gray-600 mb-1">High Risk</p>
              <p className="text-2xl font-bold text-gray-900">{metrics.high_risk_count}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ABTestDashboard;
