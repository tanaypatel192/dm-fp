import React from 'react';
import { Card } from '@/components/common';

const BatchAnalysis: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
          Batch Analysis
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Analyze multiple patients at once
        </p>
      </div>

      <Card>
        <div className="text-center py-12">
          <p className="text-gray-500 dark:text-gray-400">
            Batch analysis feature coming soon...
          </p>
        </div>
      </Card>
    </div>
  );
};

export default BatchAnalysis;
