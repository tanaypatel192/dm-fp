import React from 'react';
import {
  FiCheckCircle,
  FiAlertTriangle,
  FiAlertCircle,
  FiActivity,
  FiUsers,
  FiTrendingUp,
  FiTrendingDown,
  FiClock,
} from 'react-icons/fi';
import { Card } from '@/components/common';
import type { ComprehensivePredictionOutput } from '@/types/api';

interface PredictionResultsProps {
  result: ComprehensivePredictionOutput;
}

const PredictionResults: React.FC<PredictionResultsProps> = ({ result }) => {
  // Get risk color and icon
  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel.toLowerCase()) {
      case 'low':
        return {
          bg: 'bg-success-50 dark:bg-success-900/20',
          border: 'border-success-200 dark:border-success-800',
          text: 'text-success-700 dark:text-success-300',
          icon: <FiCheckCircle className="w-8 h-8" />,
        };
      case 'medium':
        return {
          bg: 'bg-warning-50 dark:bg-warning-900/20',
          border: 'border-warning-200 dark:border-warning-800',
          text: 'text-warning-700 dark:text-warning-300',
          icon: <FiAlertTriangle className="w-8 h-8" />,
        };
      case 'high':
        return {
          bg: 'bg-danger-50 dark:bg-danger-900/20',
          border: 'border-danger-200 dark:border-danger-800',
          text: 'text-danger-700 dark:text-danger-300',
          icon: <FiAlertCircle className="w-8 h-8" />,
        };
      default:
        return {
          bg: 'bg-gray-50 dark:bg-gray-900/20',
          border: 'border-gray-200 dark:border-gray-800',
          text: 'text-gray-700 dark:text-gray-300',
          icon: <FiActivity className="w-8 h-8" />,
        };
    }
  };

  const riskStyle = getRiskColor(result.risk_level);

  return (
    <div className="space-y-6">
      {/* Main Prediction Result */}
      <Card className={`${riskStyle.bg} ${riskStyle.border} border-2`}>
        <div className="flex items-center gap-4">
          <div className={riskStyle.text}>{riskStyle.icon}</div>
          <div className="flex-1">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              {result.ensemble_label}
            </h2>
            <p className={`text-lg font-semibold ${riskStyle.text}`}>
              Risk Level: {result.risk_level}
            </p>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-600 dark:text-gray-400">Probability</div>
            <div className={`text-3xl font-bold ${riskStyle.text}`}>
              {(result.ensemble_probability * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Confidence: {(result.ensemble_confidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        {/* Processing time */}
        <div className="flex items-center gap-2 mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <FiClock className="w-4 h-4 text-gray-500" />
          <span className="text-sm text-gray-600 dark:text-gray-400">
            Processed in {result.processing_time_ms.toFixed(0)}ms
          </span>
        </div>
      </Card>

      {/* Model Predictions */}
      <Card title="Model Predictions" subtitle="Predictions from all three ML models">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {result.model_predictions.map((model) => (
            <div
              key={model.model_name}
              className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg"
            >
              <h4 className="font-semibold text-sm text-gray-700 dark:text-gray-300 mb-2 capitalize">
                {model.model_name.replace(/_/g, ' ')}
              </h4>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-600 dark:text-gray-400">Prediction:</span>
                  <span
                    className={`font-bold text-sm ${
                      model.prediction === 1 ? 'text-danger-600' : 'text-success-600'
                    }`}
                  >
                    {model.prediction_label}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-600 dark:text-gray-400">Probability:</span>
                  <span className="font-semibold text-sm">
                    {(model.probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-600 dark:text-gray-400">Confidence:</span>
                  <span className="font-semibold text-sm">
                    {(model.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                {/* Visual bar */}
                <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden mt-2">
                  <div
                    className={`h-full ${
                      model.probability < 0.3
                        ? 'bg-success-500'
                        : model.probability < 0.7
                        ? 'bg-warning-500'
                        : 'bg-danger-500'
                    }`}
                    style={{ width: `${model.probability * 100}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* SHAP Explanation */}
      {result.shap_available && result.shap_explanation && (
        <Card title="Feature Contributions" subtitle="How each feature affects the prediction">
          <div className="space-y-3">
            {result.shap_explanation.feature_contributions
              .slice(0, 8)
              .map((contribution, idx) => {
                const isPositive = contribution.contribution > 0;
                const absContribution = Math.abs(contribution.contribution);
                const maxContribution = Math.max(
                  ...result.shap_explanation!.feature_contributions.map((c) =>
                    Math.abs(c.contribution)
                  )
                );
                const width = (absContribution / maxContribution) * 100;

                return (
                  <div key={contribution.feature}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        {contribution.feature}
                      </span>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-600 dark:text-gray-400">
                          Value: {contribution.value.toFixed(2)}
                        </span>
                        {isPositive ? (
                          <FiTrendingUp className="w-4 h-4 text-danger-600" />
                        ) : (
                          <FiTrendingDown className="w-4 h-4 text-success-600" />
                        )}
                        <span
                          className={`text-sm font-semibold ${
                            isPositive ? 'text-danger-600' : 'text-success-600'
                          }`}
                        >
                          {isPositive ? '+' : ''}
                          {contribution.contribution.toFixed(3)}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${
                            isPositive ? 'bg-danger-500' : 'bg-success-500'
                          }`}
                          style={{ width: `${width}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-500 w-16 text-right">
                        {contribution.impact}
                      </span>
                    </div>
                  </div>
                );
              })}
          </div>

          <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
            <p className="text-xs text-gray-600 dark:text-gray-400">
              <strong>Top Contributing Features:</strong>{' '}
              {result.shap_explanation.top_features.slice(0, 3).join(', ')}
            </p>
          </div>
        </Card>
      )}

      {/* Risk Factors */}
      {result.risk_factors.length > 0 && (
        <Card title="Identified Risk Factors" subtitle="Areas of concern based on your input">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {result.risk_factors.map((factor, idx) => {
              const riskColor =
                factor.risk_level.toLowerCase() === 'high'
                  ? 'border-danger-500 bg-danger-50 dark:bg-danger-900/20'
                  : 'border-warning-500 bg-warning-50 dark:bg-warning-900/20';

              return (
                <div key={idx} className={`p-4 border-2 rounded-lg ${riskColor}`}>
                  <div className="flex items-start justify-between mb-2">
                    <h4 className="font-semibold text-sm text-gray-900 dark:text-gray-100">
                      {factor.factor}
                    </h4>
                    <span
                      className={`px-2 py-1 text-xs font-semibold rounded ${
                        factor.risk_level.toLowerCase() === 'high'
                          ? 'bg-danger-100 text-danger-800 dark:bg-danger-800 dark:text-danger-100'
                          : 'bg-warning-100 text-warning-800 dark:bg-warning-800 dark:text-warning-100'
                      }`}
                    >
                      {factor.risk_level} Risk
                    </span>
                  </div>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    Current Value: <strong>{factor.current_value.toFixed(2)}</strong>
                  </p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    {factor.is_modifiable ? (
                      <span className="text-success-600 dark:text-success-400">
                        ✓ Modifiable through lifestyle changes
                      </span>
                    ) : (
                      <span className="text-gray-500">Non-modifiable factor</span>
                    )}
                  </p>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {/* Similar Patients */}
      {result.similar_patients.length > 0 && (
        <Card title="Similar Patients" subtitle="Cases from training data with similar profiles">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {result.similar_patients.map((patient, idx) => (
              <div
                key={idx}
                className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg"
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <FiUsers className="w-5 h-5 text-primary-600" />
                    <span className="text-sm font-semibold">Patient {idx + 1}</span>
                  </div>
                  <span
                    className={`px-2 py-1 text-xs font-semibold rounded ${
                      patient.outcome === 1
                        ? 'bg-danger-100 text-danger-800 dark:bg-danger-800 dark:text-danger-100'
                        : 'bg-success-100 text-success-800 dark:bg-success-800 dark:text-success-100'
                    }`}
                  >
                    {patient.outcome_label}
                  </span>
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  <p className="mb-2">
                    Similarity: <strong>{(patient.similarity_score * 100).toFixed(1)}%</strong>
                  </p>
                  {patient.key_similarities.length > 0 && (
                    <div>
                      <p className="text-xs mb-1">Similar features:</p>
                      <div className="flex flex-wrap gap-1">
                        {patient.key_similarities.map((feature) => (
                          <span
                            key={feature}
                            className="px-2 py-1 text-xs bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded"
                          >
                            {feature}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Recommendations */}
      {result.recommendations.length > 0 && (
        <Card
          title="Personalized Recommendations"
          subtitle="Action items to reduce your diabetes risk"
        >
          <div className="space-y-4">
            {result.recommendations.map((rec, idx) => {
              const priorityColor =
                rec.priority === 'High'
                  ? 'border-l-danger-500'
                  : rec.priority === 'Medium'
                  ? 'border-l-warning-500'
                  : 'border-l-primary-500';

              return (
                <div
                  key={idx}
                  className={`p-4 border-l-4 ${priorityColor} bg-gray-50 dark:bg-gray-900/50 rounded-r-lg`}
                >
                  <div className="flex items-start justify-between mb-2">
                    <h4 className="font-semibold text-gray-900 dark:text-gray-100">
                      {rec.category}
                    </h4>
                    <span
                      className={`px-2 py-1 text-xs font-semibold rounded ${
                        rec.priority === 'High'
                          ? 'bg-danger-100 text-danger-800 dark:bg-danger-800 dark:text-danger-100'
                          : rec.priority === 'Medium'
                          ? 'bg-warning-100 text-warning-800 dark:bg-warning-800 dark:text-warning-100'
                          : 'bg-primary-100 text-primary-800 dark:bg-primary-800 dark:text-primary-100'
                      }`}
                    >
                      {rec.priority} Priority
                    </span>
                  </div>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                    {rec.recommendation}
                  </p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 italic">
                    <strong>Why:</strong> {rec.rationale}
                  </p>
                </div>
              );
            })}
          </div>
        </Card>
      )}
    </div>
  );
};

export default PredictionResults;
