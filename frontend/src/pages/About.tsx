import React from 'react';
import { Card } from '@/components/common';

const About: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
          About Models
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Learn about the machine learning models used in this system
        </p>
      </div>

      <Card title="Decision Tree" className="mb-4">
        <div className="space-y-4">
          <p className="text-gray-600 dark:text-gray-400">
            A tree-based model that makes predictions by learning simple decision rules
            from the data features. Our implementation uses a comprehensive grid search
            to optimize performance.
          </p>

          <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-md">
            <h3 className="font-semibold text-sm text-gray-900 dark:text-gray-100 mb-2">
              Model Configuration & Hyperparameters
            </h3>
            <ul className="list-disc list-inside text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li><span className="font-medium">Criterion:</span> Gini Impurity, Entropy (Information Gain)</li>
              <li><span className="font-medium">Max Depth:</span> [3, 5, 7, 10, 15, 20] - Controls model complexity</li>
              <li><span className="font-medium">Min Samples Split:</span> [2, 5, 10, 15, 20] - Prevents overfitting</li>
              <li><span className="font-medium">Min Samples Leaf:</span> [1, 2, 4, 6, 8, 10] - Smooths the model</li>
            </ul>
          </div>

          <div className="text-sm text-gray-500">
            <span className="font-semibold">Key Features:</span> Automated Rule Extraction, Feature Importance Analysis,
            Prediction Path Explanation.
          </div>
        </div>
      </Card>

      <Card title="Random Forest" className="mb-4">
        <div className="space-y-4">
          <p className="text-gray-600 dark:text-gray-400">
            An ensemble method that combines multiple decision trees to improve accuracy
            and reduce overfitting. It uses bagging (bootstrap aggregating) and random
            feature selection.
          </p>

          <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-md">
            <h3 className="font-semibold text-sm text-gray-900 dark:text-gray-100 mb-2">
              Ensemble Configuration
            </h3>
            <ul className="list-disc list-inside text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li><span className="font-medium">N Estimators:</span> [50, 100, 200, 300] - Number of trees</li>
              <li><span className="font-medium">Max Features:</span> sqrt, log2, None - Random feature subset size</li>
              <li><span className="font-medium">Bootstrap:</span> True - Sampling with replacement</li>
              <li><span className="font-medium">OOB Score:</span> Enabled - Out-of-bag error estimation</li>
            </ul>
          </div>

          <div className="text-sm text-gray-500">
            <span className="font-semibold">Advanced Analysis:</span> Tree Diversity Metrics, OOB Error Analysis,
            Partial Dependence Plots.
          </div>
        </div>
      </Card>

      <Card title="XGBoost" className="mb-4">
        <div className="space-y-4">
          <p className="text-gray-600 dark:text-gray-400">
            Extreme Gradient Boosting is an optimized distributed gradient boosting library.
            It builds models sequentially, with each new model correcting the errors of the previous ones.
          </p>

          <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-md">
            <h3 className="font-semibold text-sm text-gray-900 dark:text-gray-100 mb-2">
              Gradient Boosting Parameters
            </h3>
            <ul className="list-disc list-inside text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li><span className="font-medium">Learning Rate (Eta):</span> [0.01, 0.05, 0.1, 0.3] - Step size shrinkage</li>
              <li><span className="font-medium">Max Depth:</span> [3, 5, 7, 9] - Tree depth limit</li>
              <li><span className="font-medium">Subsample:</span> [0.6, 0.8, 1.0] - Row sampling ratio</li>
              <li><span className="font-medium">Colsample By Tree:</span> [0.6, 0.8, 1.0] - Column sampling ratio</li>
              <li><span className="font-medium">Gamma:</span> [0, 0.1, 0.2] - Minimum loss reduction for split</li>
            </ul>
          </div>

          <div className="text-sm text-gray-500">
            <span className="font-semibold">Optimization:</span> GPU Acceleration (CUDA), Early Stopping,
            Weighted Quantile Sketch.
          </div>
        </div>
      </Card>

      <Card title="SHAP Explanations">
        <div className="space-y-4">
          <p className="text-gray-600 dark:text-gray-400">
            SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output
            of any machine learning model. It connects optimal credit allocation with local explanations.
          </p>

          <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-md">
            <h3 className="font-semibold text-sm text-gray-900 dark:text-gray-100 mb-2">
              Interpretability Features
            </h3>
            <ul className="list-disc list-inside text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li><span className="font-medium">Global Importance:</span> Summary plots showing feature impact across all data</li>
              <li><span className="font-medium">Local Explanation:</span> Force plots for individual predictions</li>
              <li><span className="font-medium">Dependence Plots:</span> Interaction effects between features</li>
              <li><span className="font-medium">TreeExplainer:</span> Optimized algorithm for tree-based models</li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default About;
