import React from 'react';
import { Card } from '@/components/common';

const SinglePrediction: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
          Single Prediction
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Predict diabetes risk for a single patient with detailed explanations
        </p>
      </div>

      <Card>
        <div className="text-center py-12">
          <p className="text-gray-500 dark:text-gray-400">
            Single prediction form coming soon...
          </p>
        </div>
      </Card>
    </div>
  );
};

export default SinglePrediction;
