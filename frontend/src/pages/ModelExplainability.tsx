import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import {
  FiBook,
  FiGitBranch,
  FiBarChart2,
  FiLayers,
  FiCheckCircle,
  FiAlertTriangle,
  FiZap,
  FiTrendingUp,
  FiCode,
} from 'react-icons/fi';
import { Button, Card, LoadingSpinner } from '@/components/common';
import { predictionApi, modelApi, handleApiError } from '@/services/api';
import type { PatientInput, ComprehensivePredictionOutput } from '@/types/api';

// Model information
const MODEL_INFO = {
  decision_tree: {
    name: 'Decision Tree',
    icon: <FiGitBranch className="w-6 h-6" />,
    color: '#10b981',
    description: 'A tree-like model that makes decisions by asking yes/no questions about features',
    howItWorks: [
      'Starts at the root and asks questions about feature values',
      'Each node represents a question (e.g., "Is glucose > 120?")',
      'Branches represent possible answers (Yes/No)',
      'Follows the path down the tree until reaching a leaf node',
      'Leaf node provides the final prediction',
    ],
    pros: [
      'Easy to understand and visualize',
      'Can handle both numerical and categorical data',
      'Requires little data preprocessing',
      'Can capture non-linear relationships',
      'Shows clear decision rules',
    ],
    cons: [
      'Prone to overfitting on training data',
      'Can be unstable - small data changes lead to different trees',
      'May not generalize well to new data',
      'Biased towards features with more levels',
    ],
    whenToUse: 'When interpretability is crucial and you need to explain decisions to non-technical stakeholders',
  },
  random_forest: {
    name: 'Random Forest',
    icon: <FiLayers className="w-6 h-6" />,
    color: '#3b82f6',
    description: 'An ensemble of many decision trees that vote on the final prediction',
    howItWorks: [
      'Creates hundreds of decision trees',
      'Each tree is trained on a random subset of data (bootstrap sampling)',
      'Each split considers only a random subset of features',
      'All trees make independent predictions',
      'Final prediction is the majority vote (classification) or average (regression)',
    ],
    pros: [
      'More accurate than single decision trees',
      'Reduces overfitting through averaging',
      'Handles missing data well',
      'Provides feature importance rankings',
      'Works well on large datasets',
    ],
    cons: [
      'Less interpretable than single decision tree',
      'Slower to train and predict',
      'Requires more memory',
      'Can still overfit on noisy data',
    ],
    whenToUse: 'When you need better accuracy than decision trees while maintaining some interpretability',
  },
  xgboost: {
    name: 'XGBoost',
    icon: <FiZap className="w-6 h-6" />,
    color: '#f59e0b',
    description: 'Extreme Gradient Boosting - builds trees sequentially to correct previous errors',
    howItWorks: [
      'Builds trees one at a time, sequentially',
      'Each new tree tries to correct errors made by previous trees',
      'Focuses on samples that were misclassified',
      'Uses gradient descent to minimize loss function',
      'Combines all trees with weighted sum for final prediction',
    ],
    pros: [
      'Usually the most accurate of the three',
      'Handles missing values automatically',
      'Built-in regularization to prevent overfitting',
      'Very efficient and fast',
      'Winner of many ML competitions',
    ],
    cons: [
      'Most complex and hardest to interpret',
      'Requires careful hyperparameter tuning',
      'Can overfit if not properly regularized',
      'Slower to train than Random Forest',
    ],
    whenToUse: 'When maximum accuracy is the priority and you can sacrifice some interpretability',
  },
};

// Sample patients for examples
const EXAMPLE_PATIENTS = [
  {
    name: 'Low Risk Patient',
    data: { Pregnancies: 1, Glucose: 85, BloodPressure: 66, SkinThickness: 29, Insulin: 0, BMI: 26.6, DiabetesPedigreeFunction: 0.351, Age: 31 },
    description: 'Young patient with normal glucose and BMI',
    expectedOutcome: 'Low risk - all models should agree',
  },
  {
    name: 'High Risk Patient',
    data: { Pregnancies: 6, Glucose: 148, BloodPressure: 72, SkinThickness: 35, Insulin: 0, BMI: 33.6, DiabetesPedigreeFunction: 0.627, Age: 50 },
    description: 'Older patient with high glucose and BMI',
    expectedOutcome: 'High risk - all models should agree',
  },
  {
    name: 'Borderline Patient',
    data: { Pregnancies: 3, Glucose: 120, BloodPressure: 70, SkinThickness: 30, Insulin: 100, BMI: 30.0, DiabetesPedigreeFunction: 0.4, Age: 40 },
    description: 'Middle-aged patient with borderline values',
    expectedOutcome: 'Models may disagree - interesting case',
  },
];

const ModelExplainability: React.FC = () => {
  const [activeSection, setActiveSection] = useState<'education' | 'tree' | 'importance' | 'shap' | 'examples' | 'try'>('education');
  const [selectedModel, setSelectedModel] = useState<'decision_tree' | 'random_forest' | 'xgboost'>('decision_tree');
  const [examplePredictions, setExamplePredictions] = useState<Record<string, ComprehensivePredictionOutput>>({});
  const [loadingExamples, setLoadingExamples] = useState(false);

  // Try it yourself state
  const [tryPatientData, setTryPatientData] = useState<PatientInput>({
    Pregnancies: 3,
    Glucose: 120,
    BloodPressure: 70,
    SkinThickness: 30,
    Insulin: 100,
    BMI: 30.0,
    DiabetesPedigreeFunction: 0.4,
    Age: 40,
  });
  const [tryPrediction, setTryPrediction] = useState<ComprehensivePredictionOutput | null>(null);
  const [loadingTry, setLoadingTry] = useState(false);

  // Load example predictions
  useEffect(() => {
    if (activeSection === 'examples' && Object.keys(examplePredictions).length === 0) {
      loadExamplePredictions();
    }
  }, [activeSection]);

  const loadExamplePredictions = async () => {
    setLoadingExamples(true);
    try {
      const predictions: Record<string, ComprehensivePredictionOutput> = {};
      for (const example of EXAMPLE_PATIENTS) {
        const result = await predictionApi.predictExplain(example.data);
        predictions[example.name] = result;
      }
      setExamplePredictions(predictions);
    } catch (error) {
      console.error('Error loading examples:', error);
    } finally {
      setLoadingExamples(false);
    }
  };

  const handleTryPrediction = async () => {
    setLoadingTry(true);
    try {
      const result = await predictionApi.predictExplain(tryPatientData);
      setTryPrediction(result);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoadingTry(false);
    }
  };

  // Navigation sections
  const sections = [
    { id: 'education', label: 'Learn Models', icon: <FiBook className="w-4 h-4" /> },
    { id: 'tree', label: 'Decision Trees', icon: <FiGitBranch className="w-4 h-4" /> },
    { id: 'importance', label: 'Feature Importance', icon: <FiBarChart2 className="w-4 h-4" /> },
    { id: 'shap', label: 'SHAP Values', icon: <FiTrendingUp className="w-4 h-4" /> },
    { id: 'examples', label: 'Examples', icon: <FiLayers className="w-4 h-4" /> },
    { id: 'try', label: 'Try It', icon: <FiCode className="w-4 h-4" /> },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Model Explainability</h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Understand how our ML models make predictions and interpret their decisions
        </p>
      </div>

      {/* Navigation */}
      <Card>
        <div className="flex flex-wrap gap-2">
          {sections.map((section) => (
            <button
              key={section.id}
              onClick={() => setActiveSection(section.id as any)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                activeSection === section.id
                  ? 'bg-primary-600 text-white shadow-md'
                  : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
              }`}
            >
              {section.icon}
              <span className="font-medium">{section.label}</span>
            </button>
          ))}
        </div>
      </Card>

      {/* Education Section */}
      {activeSection === 'education' && (
        <div className="space-y-6">
          {/* Model Selector */}
          <Card>
            <div className="flex gap-2 flex-wrap">
              {Object.entries(MODEL_INFO).map(([key, model]) => (
                <button
                  key={key}
                  onClick={() => setSelectedModel(key as any)}
                  className={`flex items-center gap-3 px-6 py-4 rounded-lg border-2 transition-all flex-1 min-w-[200px] ${
                    selectedModel === key
                      ? 'border-current shadow-lg scale-105'
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
                  }`}
                  style={{ color: selectedModel === key ? model.color : undefined }}
                >
                  {model.icon}
                  <span className="font-semibold">{model.name}</span>
                </button>
              ))}
            </div>
          </Card>

          {/* Model Details */}
          <Card>
            <div className="space-y-6">
              <div className="flex items-center gap-3">
                <div
                  className="w-12 h-12 rounded-lg flex items-center justify-center"
                  style={{ backgroundColor: `${MODEL_INFO[selectedModel].color}20`, color: MODEL_INFO[selectedModel].color }}
                >
                  {MODEL_INFO[selectedModel].icon}
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                    {MODEL_INFO[selectedModel].name}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-400">
                    {MODEL_INFO[selectedModel].description}
                  </p>
                </div>
              </div>

              {/* How it works */}
              <div>
                <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3">
                  How It Works
                </h4>
                <ol className="space-y-2">
                  {MODEL_INFO[selectedModel].howItWorks.map((step, index) => (
                    <li key={index} className="flex gap-3">
                      <span
                        className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold text-white"
                        style={{ backgroundColor: MODEL_INFO[selectedModel].color }}
                      >
                        {index + 1}
                      </span>
                      <span className="text-gray-700 dark:text-gray-300">{step}</span>
                    </li>
                  ))}
                </ol>
              </div>

              {/* Pros and Cons */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3 flex items-center gap-2">
                    <FiCheckCircle className="text-success-600" />
                    Advantages
                  </h4>
                  <ul className="space-y-2">
                    {MODEL_INFO[selectedModel].pros.map((pro, index) => (
                      <li key={index} className="flex gap-2 text-gray-700 dark:text-gray-300">
                        <span className="text-success-600 mt-0.5">✓</span>
                        <span>{pro}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3 flex items-center gap-2">
                    <FiAlertTriangle className="text-warning-600" />
                    Limitations
                  </h4>
                  <ul className="space-y-2">
                    {MODEL_INFO[selectedModel].cons.map((con, index) => (
                      <li key={index} className="flex gap-2 text-gray-700 dark:text-gray-300">
                        <span className="text-warning-600 mt-0.5">⚠</span>
                        <span>{con}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* When to use */}
              <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
                <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
                  When to Use This Model
                </h4>
                <p className="text-blue-800 dark:text-blue-200">
                  {MODEL_INFO[selectedModel].whenToUse}
                </p>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Decision Tree Visualization */}
      {activeSection === 'tree' && (
        <div className="space-y-6">
          <Card>
            <div className="space-y-4">
              <div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-2">
                  Decision Tree Visualization
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  Simplified view of how a decision tree makes decisions
                </p>
              </div>

              {/* Simplified tree diagram */}
              <div className="p-8 bg-gray-50 dark:bg-gray-800 rounded-lg overflow-x-auto">
                <div className="inline-block min-w-full">
                  <div className="flex flex-col items-center space-y-4">
                    {/* Root node */}
                    <div className="bg-blue-100 dark:bg-blue-900/30 border-2 border-blue-500 rounded-lg p-4 text-center">
                      <div className="font-bold text-blue-900 dark:text-blue-100">Root</div>
                      <div className="text-sm text-blue-700 dark:text-blue-300">Glucose {'>'} 120?</div>
                    </div>

                    {/* Branches */}
                    <div className="flex gap-32">
                      <div className="flex flex-col items-center space-y-4">
                        <div className="text-sm font-medium text-gray-600 dark:text-gray-400">Yes</div>
                        <div className="bg-orange-100 dark:bg-orange-900/30 border-2 border-orange-500 rounded-lg p-4 text-center">
                          <div className="font-bold text-orange-900 dark:text-orange-100">Node</div>
                          <div className="text-sm text-orange-700 dark:text-orange-300">BMI {'>'} 30?</div>
                        </div>
                        <div className="flex gap-12">
                          <div className="flex flex-col items-center space-y-2">
                            <div className="text-xs font-medium text-gray-600 dark:text-gray-400">Yes</div>
                            <div className="bg-red-100 dark:bg-red-900/30 border-2 border-red-500 rounded-lg p-3 text-center">
                              <div className="font-bold text-red-900 dark:text-red-100 text-sm">Diabetic</div>
                              <div className="text-xs text-red-700 dark:text-red-300">85% confidence</div>
                            </div>
                          </div>
                          <div className="flex flex-col items-center space-y-2">
                            <div className="text-xs font-medium text-gray-600 dark:text-gray-400">No</div>
                            <div className="bg-yellow-100 dark:bg-yellow-900/30 border-2 border-yellow-500 rounded-lg p-3 text-center">
                              <div className="font-bold text-yellow-900 dark:text-yellow-100 text-sm">Check Age</div>
                              <div className="text-xs text-yellow-700 dark:text-yellow-300">Age {'>'} 45?</div>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="flex flex-col items-center space-y-4">
                        <div className="text-sm font-medium text-gray-600 dark:text-gray-400">No</div>
                        <div className="bg-green-100 dark:bg-green-900/30 border-2 border-green-500 rounded-lg p-4 text-center">
                          <div className="font-bold text-green-900 dark:text-green-100">Non-Diabetic</div>
                          <div className="text-sm text-green-700 dark:text-green-300">90% confidence</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Explanation */}
              <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">
                  How to Read This Tree
                </h4>
                <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                  <li className="flex gap-2">
                    <span className="text-blue-600">●</span>
                    <span><strong>Decision nodes</strong> (rectangles) contain questions about features</span>
                  </li>
                  <li className="flex gap-2">
                    <span className="text-green-600">●</span>
                    <span><strong>Leaf nodes</strong> (bottom nodes) contain final predictions</span>
                  </li>
                  <li className="flex gap-2">
                    <span className="text-purple-600">●</span>
                    <span><strong>Branches</strong> show the path taken based on Yes/No answers</span>
                  </li>
                  <li className="flex gap-2">
                    <span className="text-orange-600">●</span>
                    <span><strong>Confidence</strong> shows how certain the model is about its prediction</span>
                  </li>
                </ul>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Feature Importance */}
      {activeSection === 'importance' && (
        <div className="space-y-6">
          <Card>
            <div className="space-y-4">
              <div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-2">
                  Feature Importance Comparison
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  See which features each model considers most important
                </p>
              </div>

              {/* Mock feature importance chart */}
              <Plot
                data={[
                  {
                    x: [0.35, 0.25, 0.15, 0.10, 0.08, 0.04, 0.02, 0.01],
                    y: ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'BloodPressure', 'Pregnancies', 'Insulin', 'SkinThickness'],
                    type: 'bar',
                    orientation: 'h',
                    name: 'Decision Tree',
                    marker: { color: '#10b981' },
                  },
                  {
                    x: [0.32, 0.28, 0.18, 0.09, 0.06, 0.04, 0.02, 0.01],
                    y: ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'BloodPressure', 'Pregnancies', 'Insulin', 'SkinThickness'],
                    type: 'bar',
                    orientation: 'h',
                    name: 'Random Forest',
                    marker: { color: '#3b82f6' },
                  },
                  {
                    x: [0.30, 0.26, 0.20, 0.11, 0.07, 0.03, 0.02, 0.01],
                    y: ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'BloodPressure', 'Pregnancies', 'Insulin', 'SkinThickness'],
                    type: 'bar',
                    orientation: 'h',
                    name: 'XGBoost',
                    marker: { color: '#f59e0b' },
                  },
                ]}
                layout={{
                  title: 'Feature Importance Across Models',
                  xaxis: { title: 'Importance Score' },
                  barmode: 'group',
                  height: 500,
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                }}
                config={{ responsive: true }}
                style={{ width: '100%' }}
              />

              {/* Explanation */}
              <div className="space-y-4">
                <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
                  <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
                    What is Feature Importance?
                  </h4>
                  <p className="text-sm text-blue-800 dark:text-blue-200">
                    Feature importance measures how much each feature contributes to the model's predictions.
                    Higher values mean the feature is more influential in determining the outcome.
                  </p>
                </div>

                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg">
                  <h4 className="font-semibold text-purple-900 dark:text-purple-100 mb-2">
                    Why Do Models Rank Features Differently?
                  </h4>
                  <ul className="text-sm text-purple-800 dark:text-purple-200 space-y-2">
                    <li>• <strong>Different algorithms</strong> calculate importance in different ways</li>
                    <li>• <strong>Decision Trees</strong> use information gain at splits</li>
                    <li>• <strong>Random Forest</strong> averages importance across many trees</li>
                    <li>• <strong>XGBoost</strong> uses gradient-based importance metrics</li>
                    <li>• Some features may be important only in combination with others</li>
                  </ul>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* SHAP Values */}
      {activeSection === 'shap' && (
        <div className="space-y-6">
          <Card>
            <div className="space-y-4">
              <div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-2">
                  Understanding SHAP Values
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  SHAP (SHapley Additive exPlanations) explains individual predictions
                </p>
              </div>

              {/* SHAP explanation cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg">
                  <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
                    What are SHAP Values?
                  </h4>
                  <p className="text-sm text-blue-800 dark:text-blue-200">
                    SHAP values show how much each feature contributed to moving the prediction away from
                    the baseline (average) prediction. Positive values push toward "Diabetic", negative
                    values push toward "Non-Diabetic".
                  </p>
                </div>

                <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg">
                  <h4 className="font-semibold text-green-900 dark:text-green-100 mb-2">
                    How to Interpret
                  </h4>
                  <ul className="text-sm text-green-800 dark:text-green-200 space-y-1">
                    <li>• <strong>Red bars</strong>: Features pushing toward Diabetic</li>
                    <li>• <strong>Green bars</strong>: Features pushing toward Non-Diabetic</li>
                    <li>• <strong>Longer bars</strong>: Stronger influence</li>
                    <li>• <strong>Base value</strong>: Average prediction across all patients</li>
                  </ul>
                </div>
              </div>

              {/* Sample SHAP waterfall */}
              <div>
                <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">
                  Example SHAP Waterfall Plot
                </h4>
                <Plot
                  data={[
                    {
                      x: [0.35, 0.20, -0.15, 0.10, -0.08, 0.05, -0.03, 0.01],
                      y: ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'BloodPressure', 'Pregnancies', 'Insulin', 'SkinThickness'],
                      type: 'bar',
                      orientation: 'h',
                      marker: {
                        color: [
                          'rgba(239, 68, 68, 0.7)',
                          'rgba(239, 68, 68, 0.7)',
                          'rgba(34, 197, 94, 0.7)',
                          'rgba(239, 68, 68, 0.7)',
                          'rgba(34, 197, 94, 0.7)',
                          'rgba(239, 68, 68, 0.7)',
                          'rgba(34, 197, 94, 0.7)',
                          'rgba(239, 68, 68, 0.7)',
                        ],
                      },
                    },
                  ]}
                  layout={{
                    title: 'Feature Contributions (SHAP Values)',
                    xaxis: { title: 'Impact on Prediction', zeroline: true },
                    height: 400,
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                  }}
                  config={{ responsive: true }}
                  style={{ width: '100%' }}
                />
              </div>

              {/* Interpretation guide */}
              <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">
                  Reading the Plot Above
                </h4>
                <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                  <li>• <strong>Glucose</strong> (+0.35): High glucose strongly pushes toward diabetic prediction</li>
                  <li>• <strong>BMI</strong> (+0.20): High BMI moderately increases risk</li>
                  <li>• <strong>Age</strong> (-0.15): Younger age reduces predicted risk</li>
                  <li>• The model combines all these factors to reach its final prediction</li>
                </ul>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Examples */}
      {activeSection === 'examples' && (
        <div className="space-y-6">
          {loadingExamples ? (
            <Card>
              <div className="text-center py-12">
                <LoadingSpinner size="lg" />
                <p className="text-gray-600 dark:text-gray-400 mt-4">Loading example predictions...</p>
              </div>
            </Card>
          ) : (
            EXAMPLE_PATIENTS.map((example, index) => (
              <Card key={index}>
                <div className="space-y-4">
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100">
                        {example.name}
                      </h3>
                      <p className="text-gray-600 dark:text-gray-400 mt-1">{example.description}</p>
                      <p className="text-sm text-blue-600 dark:text-blue-400 mt-1">
                        Expected: {example.expectedOutcome}
                      </p>
                    </div>
                  </div>

                  {/* Patient data */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {Object.entries(example.data).map(([key, value]) => (
                      <div key={key} className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                        <div className="text-xs text-gray-600 dark:text-gray-400">{key}</div>
                        <div className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                          {typeof value === 'number' ? value.toFixed(2) : value}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Predictions */}
                  {examplePredictions[example.name] && (
                    <div className="space-y-3">
                      <h4 className="font-semibold text-gray-900 dark:text-gray-100">Model Predictions</h4>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        {examplePredictions[example.name].model_predictions.map((pred) => (
                          <div
                            key={pred.model_name}
                            className={`p-4 rounded-lg border-2 ${
                              pred.prediction === 1
                                ? 'bg-danger-50 dark:bg-danger-900/20 border-danger-300 dark:border-danger-700'
                                : 'bg-success-50 dark:bg-success-900/20 border-success-300 dark:border-success-700'
                            }`}
                          >
                            <div className="font-semibold capitalize">
                              {pred.model_name.replace('_', ' ')}
                            </div>
                            <div className="text-2xl font-bold mt-1">
                              {pred.prediction_label}
                            </div>
                            <div className="text-sm mt-1">
                              {(pred.probability * 100).toFixed(1)}% confidence
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </Card>
            ))
          )}
        </div>
      )}

      {/* Try It Yourself */}
      {activeSection === 'try' && (
        <div className="space-y-6">
          <Card>
            <div className="space-y-4">
              <div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100 mb-2">
                  Try It Yourself
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  Enter patient data and see how different models explain their predictions
                </p>
              </div>

              {/* Input form */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(tryPatientData).map(([key, value]) => (
                  <div key={key}>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      {key}
                    </label>
                    <input
                      type="number"
                      value={value}
                      onChange={(e) =>
                        setTryPatientData({ ...tryPatientData, [key]: parseFloat(e.target.value) || 0 })
                      }
                      className="input w-full"
                      step="0.01"
                    />
                  </div>
                ))}
              </div>

              <Button
                variant="primary"
                onClick={handleTryPrediction}
                loading={loadingTry}
                disabled={loadingTry}
                className="w-full md:w-auto"
              >
                Get Predictions & Explanations
              </Button>

              {/* Results */}
              {tryPrediction && (
                <div className="space-y-4 mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
                  <h4 className="font-semibold text-gray-900 dark:text-gray-100">Results</h4>

                  {/* Model predictions */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {tryPrediction.model_predictions.map((pred) => (
                      <div
                        key={pred.model_name}
                        className={`p-4 rounded-lg border-2 ${
                          pred.prediction === 1
                            ? 'bg-danger-50 dark:bg-danger-900/20 border-danger-300'
                            : 'bg-success-50 dark:bg-success-900/20 border-success-300'
                        }`}
                      >
                        <div className="font-semibold capitalize">
                          {pred.model_name.replace('_', ' ')}
                        </div>
                        <div className="text-2xl font-bold mt-1">{pred.prediction_label}</div>
                        <div className="text-sm mt-1">{(pred.probability * 100).toFixed(1)}%</div>
                      </div>
                    ))}
                  </div>

                  {/* SHAP explanation */}
                  {tryPrediction.shap_explanation && (
                    <div>
                      <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-3">
                        Feature Contributions (SHAP)
                      </h4>
                      <Plot
                        data={[
                          {
                            x: tryPrediction.shap_explanation.feature_contributions.map((f) => f.contribution),
                            y: tryPrediction.shap_explanation.feature_contributions.map((f) => f.feature),
                            type: 'bar',
                            orientation: 'h',
                            marker: {
                              color: tryPrediction.shap_explanation.feature_contributions.map((f) =>
                                f.contribution > 0 ? 'rgba(239, 68, 68, 0.7)' : 'rgba(34, 197, 94, 0.7)'
                              ),
                            },
                          },
                        ]}
                        layout={{
                          xaxis: { title: 'Impact on Prediction', zeroline: true },
                          height: 400,
                          paper_bgcolor: 'rgba(0,0,0,0)',
                          plot_bgcolor: 'rgba(0,0,0,0)',
                        }}
                        config={{ responsive: true }}
                        style={{ width: '100%' }}
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};

export default ModelExplainability;
