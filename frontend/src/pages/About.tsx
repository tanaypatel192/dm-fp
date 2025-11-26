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
        <p className="text-gray-600 dark:text-gray-400">
          A tree-based model that makes predictions by learning simple decision rules
          from the data features. Easy to interpret and visualize.
        </p>
      </Card>

      <Card title="Random Forest" className="mb-4">
        <p className="text-gray-600 dark:text-gray-400">
          An ensemble method that combines multiple decision trees to improve accuracy
          and reduce overfitting. Provides robust predictions.
        </p>
      </Card>

      <Card title="XGBoost" className="mb-4">
        <p className="text-gray-600 dark:text-gray-400">
          An advanced gradient boosting algorithm that builds models sequentially,
          with each new model correcting errors from previous ones. Often provides
          the best performance.
        </p>
      </Card>

      <Card title="SHAP Explanations">
        <p className="text-gray-600 dark:text-gray-400">
          SHAP (SHapley Additive exPlanations) values help explain individual predictions
          by showing how each feature contributes to the final prediction. This makes
          the models more transparent and trustworthy.
        </p>
      </Card>
    </div>
  );
};

export default About;
