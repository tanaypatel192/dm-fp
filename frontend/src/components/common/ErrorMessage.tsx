import React from 'react';
import { FiAlertCircle, FiX } from 'react-icons/fi';

interface ErrorMessageProps {
  message: string;
  title?: string;
  onDismiss?: () => void;
  variant?: 'error' | 'warning' | 'info';
}

const ErrorMessage: React.FC<ErrorMessageProps> = ({
  message,
  title,
  onDismiss,
  variant = 'error',
}) => {
  const variantStyles = {
    error: {
      container: 'bg-danger-50 dark:bg-danger-900/20 border-danger-200 dark:border-danger-800',
      icon: 'text-danger-600 dark:text-danger-400',
      title: 'text-danger-800 dark:text-danger-300',
      message: 'text-danger-700 dark:text-danger-400',
    },
    warning: {
      container: 'bg-warning-50 dark:bg-warning-900/20 border-warning-200 dark:border-warning-800',
      icon: 'text-warning-600 dark:text-warning-400',
      title: 'text-warning-800 dark:text-warning-300',
      message: 'text-warning-700 dark:text-warning-400',
    },
    info: {
      container: 'bg-primary-50 dark:bg-primary-900/20 border-primary-200 dark:border-primary-800',
      icon: 'text-primary-600 dark:text-primary-400',
      title: 'text-primary-800 dark:text-primary-300',
      message: 'text-primary-700 dark:text-primary-400',
    },
  };

  const styles = variantStyles[variant];

  return (
    <div className={`border rounded-lg p-4 ${styles.container}`}>
      <div className="flex items-start gap-3">
        <FiAlertCircle className={`flex-shrink-0 w-5 h-5 mt-0.5 ${styles.icon}`} />
        <div className="flex-1">
          {title && (
            <h3 className={`font-semibold mb-1 ${styles.title}`}>{title}</h3>
          )}
          <p className={`text-sm ${styles.message}`}>{message}</p>
        </div>
        {onDismiss && (
          <button
            onClick={onDismiss}
            className={`flex-shrink-0 ${styles.icon} hover:opacity-70 transition-opacity`}
            aria-label="Dismiss"
          >
            <FiX className="w-5 h-5" />
          </button>
        )}
      </div>
    </div>
  );
};

export default ErrorMessage;
