import React from 'react';
import {
  FiInbox,
  FiFileText,
  FiUsers,
  FiSearch,
  FiAlertCircle,
  FiDatabase,
  FiFilter,
  FiPlus,
  FiUpload,
} from 'react-icons/fi';

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
    variant?: 'primary' | 'secondary';
  };
  secondaryAction?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

/**
 * EmptyState Component
 *
 * Displays a message when no data is available with optional call-to-action
 */
export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  action,
  secondaryAction,
  className = '',
  size = 'md',
}) => {
  const sizes = {
    sm: {
      container: 'py-8',
      icon: 'w-12 h-12',
      title: 'text-lg',
      description: 'text-sm',
    },
    md: {
      container: 'py-12',
      icon: 'w-16 h-16',
      title: 'text-xl',
      description: 'text-base',
    },
    lg: {
      container: 'py-16',
      icon: 'w-20 h-20',
      title: 'text-2xl',
      description: 'text-lg',
    },
  };

  return (
    <div className={`text-center ${sizes[size].container} ${className}`}>
      {/* Icon */}
      {icon && (
        <div className="flex justify-center mb-4">
          <div className="p-4 bg-gray-100 dark:bg-gray-800 rounded-full">
            <div className={`${sizes[size].icon} text-gray-400 dark:text-gray-600`}>
              {icon}
            </div>
          </div>
        </div>
      )}

      {/* Title */}
      <h3 className={`${sizes[size].title} font-semibold text-gray-900 dark:text-gray-100 mb-2`}>
        {title}
      </h3>

      {/* Description */}
      {description && (
        <p className={`${sizes[size].description} text-gray-600 dark:text-gray-400 mb-6 max-w-md mx-auto`}>
          {description}
        </p>
      )}

      {/* Actions */}
      {(action || secondaryAction) && (
        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          {action && (
            <button
              onClick={action.onClick}
              className={`px-6 py-2.5 rounded-lg font-medium transition-colors ${
                action.variant === 'secondary'
                  ? 'bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600'
                  : 'bg-primary-600 text-white hover:bg-primary-700'
              }`}
            >
              {action.label}
            </button>
          )}
          {secondaryAction && (
            <button
              onClick={secondaryAction.onClick}
              className="px-6 py-2.5 rounded-lg font-medium bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
            >
              {secondaryAction.label}
            </button>
          )}
        </div>
      )}
    </div>
  );
};

/**
 * Empty State Presets for common scenarios
 */

export const EmptyStateNoData: React.FC<{
  title?: string;
  description?: string;
  onRefresh?: () => void;
}> = ({
  title = 'No Data Available',
  description = 'There is no data to display at this time.',
  onRefresh,
}) => (
  <EmptyState
    icon={<FiInbox className="w-full h-full" />}
    title={title}
    description={description}
    action={
      onRefresh
        ? {
            label: 'Refresh',
            onClick: onRefresh,
            variant: 'secondary',
          }
        : undefined
    }
  />
);

export const EmptyStateNoPredictions: React.FC<{
  onStartPrediction?: () => void;
}> = ({ onStartPrediction }) => (
  <EmptyState
    icon={<FiFileText className="w-full h-full" />}
    title="No Predictions Yet"
    description="Start by entering patient data to get a diabetes risk prediction."
    action={
      onStartPrediction
        ? {
            label: 'Start Prediction',
            onClick: onStartPrediction,
          }
        : undefined
    }
  />
);

export const EmptyStateNoPatients: React.FC<{
  onAddPatient?: () => void;
  onUploadBatch?: () => void;
}> = ({ onAddPatient, onUploadBatch }) => (
  <EmptyState
    icon={<FiUsers className="w-full h-full" />}
    title="No Patients Found"
    description="Add patient data to start analyzing diabetes risk."
    action={
      onAddPatient
        ? {
            label: 'Add Patient',
            onClick: onAddPatient,
          }
        : undefined
    }
    secondaryAction={
      onUploadBatch
        ? {
            label: 'Upload Batch',
            onClick: onUploadBatch,
          }
        : undefined
    }
  />
);

export const EmptyStateSearchResults: React.FC<{
  searchTerm?: string;
  onClearSearch?: () => void;
}> = ({ searchTerm, onClearSearch }) => (
  <EmptyState
    icon={<FiSearch className="w-full h-full" />}
    title="No Results Found"
    description={
      searchTerm
        ? `No results found for "${searchTerm}". Try adjusting your search.`
        : 'No results found. Try adjusting your filters.'
    }
    action={
      onClearSearch
        ? {
            label: 'Clear Search',
            onClick: onClearSearch,
            variant: 'secondary',
          }
        : undefined
    }
  />
);

export const EmptyStateError: React.FC<{
  title?: string;
  description?: string;
  onRetry?: () => void;
}> = ({
  title = 'Unable to Load Data',
  description = 'An error occurred while loading data. Please try again.',
  onRetry,
}) => (
  <EmptyState
    icon={<FiAlertCircle className="w-full h-full" />}
    title={title}
    description={description}
    action={
      onRetry
        ? {
            label: 'Retry',
            onClick: onRetry,
          }
        : undefined
    }
  />
);

export const EmptyStateNoRecords: React.FC<{
  entityName?: string;
  onCreate?: () => void;
}> = ({ entityName = 'records', onCreate }) => (
  <EmptyState
    icon={<FiDatabase className="w-full h-full" />}
    title={`No ${entityName} Yet`}
    description={`You haven't created any ${entityName} yet. Get started by creating your first one.`}
    action={
      onCreate
        ? {
            label: `Create ${entityName}`,
            onClick: onCreate,
          }
        : undefined
    }
  />
);

export const EmptyStateFiltered: React.FC<{
  onClearFilters?: () => void;
}> = ({ onClearFilters }) => (
  <EmptyState
    icon={<FiFilter className="w-full h-full" />}
    title="No Matching Results"
    description="No items match your current filters. Try adjusting or clearing your filters."
    action={
      onClearFilters
        ? {
            label: 'Clear Filters',
            onClick: onClearFilters,
            variant: 'secondary',
          }
        : undefined
    }
  />
);

export const EmptyStateBatchUpload: React.FC<{
  onUpload?: () => void;
}> = ({ onUpload }) => (
  <EmptyState
    icon={<FiUpload className="w-full h-full" />}
    title="Upload Patient Data"
    description="Upload a CSV file containing patient information to perform batch analysis."
    action={
      onUpload
        ? {
            label: 'Upload CSV',
            onClick: onUpload,
          }
        : undefined
    }
  />
);

export const EmptyStateCreateFirst: React.FC<{
  title: string;
  description: string;
  actionLabel: string;
  onAction: () => void;
}> = ({ title, description, actionLabel, onAction }) => (
  <EmptyState
    icon={<FiPlus className="w-full h-full" />}
    title={title}
    description={description}
    action={{
      label: actionLabel,
      onClick: onAction,
    }}
  />
);

/**
 * Empty State with Illustration
 */
export const EmptyStateIllustration: React.FC<{
  illustration?: React.ReactNode;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}> = ({ illustration, title, description, action }) => (
  <div className="text-center py-12">
    {illustration && (
      <div className="flex justify-center mb-6">
        {illustration}
      </div>
    )}
    <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
      {title}
    </h3>
    {description && (
      <p className="text-base text-gray-600 dark:text-gray-400 mb-6 max-w-md mx-auto">
        {description}
      </p>
    )}
    {action && (
      <button
        onClick={action.onClick}
        className="px-6 py-2.5 rounded-lg font-medium bg-primary-600 text-white hover:bg-primary-700 transition-colors"
      >
        {action.label}
      </button>
    )}
  </div>
);

/**
 * Mini Empty State for smaller containers
 */
export const MiniEmptyState: React.FC<{
  message: string;
  icon?: React.ReactNode;
}> = ({ message, icon }) => (
  <div className="flex flex-col items-center justify-center py-8 text-center">
    {icon && (
      <div className="w-8 h-8 text-gray-400 dark:text-gray-600 mb-2">
        {icon}
      </div>
    )}
    <p className="text-sm text-gray-600 dark:text-gray-400">{message}</p>
  </div>
);

export default EmptyState;
