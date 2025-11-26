import React, { useState } from 'react';
import { FiInfo, FiBarChart2, FiSliders } from 'react-icons/fi';
import { Card } from '@/components/common';
import PredictionForm from '@/components/PredictionForm';
import PredictionResults from '@/components/PredictionResults';
import FeatureExplorer from '@/components/FeatureExplorer';
import type { ComprehensivePredictionOutput } from '@/types/api';

const SinglePrediction: React.FC = () => {
  const [predictionResult, setPredictionResult] =
    useState<ComprehensivePredictionOutput | null>(null);
  const [showResults, setShowResults] = useState(false);
  const [activeTab, setActiveTab] = useState<'results' | 'explorer'>('results');

  const handlePredictionComplete = (result: ComprehensivePredictionOutput) => {
    setPredictionResult(result);
    setShowResults(true);

    // Scroll to results
    setTimeout(() => {
      document.getElementById('results-section')?.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      });
    }, 100);
  };

  const handlePredictionStart = () => {
    setShowResults(false);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
          Single Prediction
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Enter patient information to predict diabetes risk with detailed explanations
        </p>
      </div>

      {/* Info Banner */}
      <Card>
        <div className="flex items-start gap-3">
          <FiInfo className="w-5 h-5 text-primary-600 dark:text-primary-400 mt-0.5 flex-shrink-0" />
          <div className="space-y-2">
            <h4 className="font-semibold text-gray-900 dark:text-gray-100">
              How This Works
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              This tool uses three machine learning models (Decision Tree, Random Forest, and
              XGBoost) to predict diabetes risk. The prediction is based on 8 clinical features
              commonly used in diabetes screening.
            </p>
            <ul className="text-sm text-gray-600 dark:text-gray-400 list-disc list-inside space-y-1">
              <li>
                <strong>Ensemble Prediction:</strong> Combines predictions from all three models
                for better accuracy
              </li>
              <li>
                <strong>SHAP Explanations:</strong> Shows which features contribute most to the
                prediction
              </li>
              <li>
                <strong>Risk Assessment:</strong> Identifies modifiable and non-modifiable risk
                factors
              </li>
              <li>
                <strong>Personalized Recommendations:</strong> Provides actionable health advice
                based on your specific risk factors
              </li>
              <li>
                <strong>Similar Cases:</strong> Compares your profile with similar patients from
                training data
              </li>
            </ul>
          </div>
        </div>
      </Card>

      {/* Prediction Form */}
      <Card title="Patient Information" subtitle="Enter all required health metrics">
        <PredictionForm
          onPredictionComplete={handlePredictionComplete}
          onPredictionStart={handlePredictionStart}
        />
      </Card>

      {/* Results Section with Tabs */}
      {showResults && predictionResult && (
        <div id="results-section" className="scroll-mt-6 space-y-4">
          {/* Header with Tabs */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  Prediction Analysis
                </h2>
                <p className="text-gray-600 dark:text-gray-400 mt-1">
                  {activeTab === 'results'
                    ? 'Comprehensive analysis with model predictions and recommendations'
                    : 'Interactive feature exploration and what-if scenarios'}
                </p>
              </div>
            </div>

            {/* Tab Navigation */}
            <div className="flex gap-2 border-b border-gray-200 dark:border-gray-700">
              <button
                onClick={() => setActiveTab('results')}
                className={`px-4 py-2 font-medium text-sm transition-colors relative ${
                  activeTab === 'results'
                    ? 'text-primary-600 dark:text-primary-400'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100'
                }`}
              >
                <div className="flex items-center gap-2">
                  <FiBarChart2 className="w-4 h-4" />
                  <span>Results & Insights</span>
                </div>
                {activeTab === 'results' && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-600 dark:bg-primary-400" />
                )}
              </button>
              <button
                onClick={() => setActiveTab('explorer')}
                className={`px-4 py-2 font-medium text-sm transition-colors relative ${
                  activeTab === 'explorer'
                    ? 'text-primary-600 dark:text-primary-400'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100'
                }`}
              >
                <div className="flex items-center gap-2">
                  <FiSliders className="w-4 h-4" />
                  <span>Interactive Explorer</span>
                </div>
                {activeTab === 'explorer' && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary-600 dark:bg-primary-400" />
                )}
              </button>
            </div>
          </div>

          {/* Tab Content */}
          <div className="min-h-[400px]">
            {activeTab === 'results' ? (
              <PredictionResults result={predictionResult} />
            ) : (
              <FeatureExplorer initialData={predictionResult.input_data} />
            )}
          </div>
        </div>
      )}

      {/* Empty State */}
      {!showResults && (
        <Card className="border-2 border-dashed border-gray-300 dark:border-gray-700">
          <div className="text-center py-12">
            <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center">
              <FiInfo className="w-8 h-8 text-gray-400" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
              No Prediction Yet
            </h3>
            <p className="text-gray-600 dark:text-gray-400 max-w-md mx-auto">
              Fill in the patient information above and click "Predict Diabetes Risk" to see
              comprehensive results with model predictions, SHAP explanations, and personalized
              recommendations.
            </p>
          </div>
        </Card>
      )}

      {/* Additional Info */}
      <Card>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
              Risk Levels
            </h4>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-success-500 rounded-full" />
                <span className="text-gray-600 dark:text-gray-400">
                  Low Risk: &lt;30% probability
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-warning-500 rounded-full" />
                <span className="text-gray-600 dark:text-gray-400">
                  Medium Risk: 30-70% probability
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-danger-500 rounded-full" />
                <span className="text-gray-600 dark:text-gray-400">
                  High Risk: &gt;70% probability
                </span>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
              Model Accuracy
            </h4>
            <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
              <p>Decision Tree: ~75-80%</p>
              <p>Random Forest: ~80-85%</p>
              <p>XGBoost: ~85-90%</p>
              <p className="text-xs italic mt-2">
                Ensemble combines all models for best accuracy
              </p>
            </div>
          </div>

          <div>
            <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
              Important Notes
            </h4>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400 list-disc list-inside">
              <li>This is a screening tool, not a diagnosis</li>
              <li>Always consult healthcare professionals</li>
              <li>Results based on training data patterns</li>
              <li>Individual results may vary</li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default SinglePrediction;
