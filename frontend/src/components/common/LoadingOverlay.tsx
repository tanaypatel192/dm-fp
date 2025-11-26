import React from 'react';
import { FiLoader } from 'react-icons/fi';

interface LoadingOverlayProps {
  isLoading: boolean;
  message?: string;
  progress?: number;
  fullScreen?: boolean;
  transparent?: boolean;
  children?: React.ReactNode;
}

/**
 * LoadingOverlay Component
 *
 * Full-screen or container-level loading overlay with optional progress bar
 */
export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  isLoading,
  message = 'Loading...',
  progress,
  fullScreen = false,
  transparent = false,
  children,
}) => {
  if (!isLoading && !children) return null;

  const overlayClasses = fullScreen
    ? 'fixed inset-0 z-50'
    : 'absolute inset-0 z-10';

  const bgClasses = transparent
    ? 'bg-white/50 dark:bg-gray-900/50'
    : 'bg-white/90 dark:bg-gray-900/90';

  return (
    <>
      {children}
      {isLoading && (
        <div className={`${overlayClasses} ${bgClasses} backdrop-blur-sm flex items-center justify-center`}>
          <div className="text-center">
            {/* Spinner */}
            <div className="inline-flex items-center justify-center w-16 h-16 mb-4">
              <FiLoader className="w-12 h-12 text-primary-600 animate-spin" />
            </div>

            {/* Message */}
            {message && (
              <div className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
                {message}
              </div>
            )}

            {/* Progress Bar */}
            {typeof progress === 'number' && (
              <div className="w-64 mx-auto">
                <div className="flex justify-between text-sm text-gray-600 dark:text-gray-400 mb-2">
                  <span>Progress</span>
                  <span>{Math.round(progress)}%</span>
                </div>
                <div className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary-600 transition-all duration-300 ease-out"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
};

/**
 * Inline Loading Component
 */
export const InlineLoading: React.FC<{
  size?: 'sm' | 'md' | 'lg';
  message?: string;
  className?: string;
}> = ({ size = 'md', message, className = '' }) => {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
  };

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <FiLoader className={`${sizes[size]} text-primary-600 animate-spin flex-shrink-0`} />
      {message && (
        <span className="text-gray-700 dark:text-gray-300">{message}</span>
      )}
    </div>
  );
};

/**
 * Button Loading State
 */
export const ButtonLoading: React.FC<{
  isLoading: boolean;
  children: React.ReactNode;
  loadingText?: string;
  className?: string;
  disabled?: boolean;
  onClick?: () => void;
  type?: 'button' | 'submit' | 'reset';
}> = ({
  isLoading,
  children,
  loadingText,
  className = '',
  disabled = false,
  onClick,
  type = 'button',
}) => {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={isLoading || disabled}
      className={`relative ${className} ${isLoading || disabled ? 'opacity-70 cursor-not-allowed' : ''}`}
    >
      {isLoading && (
        <span className="absolute inset-0 flex items-center justify-center">
          <FiLoader className="w-5 h-5 animate-spin" />
        </span>
      )}
      <span className={isLoading ? 'invisible' : ''}>
        {isLoading && loadingText ? loadingText : children}
      </span>
    </button>
  );
};

/**
 * Progress Bar Component
 */
export const ProgressBar: React.FC<{
  progress: number;
  label?: string;
  showPercentage?: boolean;
  color?: 'primary' | 'success' | 'warning' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}> = ({
  progress,
  label,
  showPercentage = true,
  color = 'primary',
  size = 'md',
  className = '',
}) => {
  const colorClasses = {
    primary: 'bg-primary-600',
    success: 'bg-success-600',
    warning: 'bg-warning-600',
    danger: 'bg-danger-600',
  };

  const sizeClasses = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3',
  };

  const clampedProgress = Math.max(0, Math.min(100, progress));

  return (
    <div className={className}>
      {(label || showPercentage) && (
        <div className="flex justify-between text-sm text-gray-700 dark:text-gray-300 mb-2">
          {label && <span>{label}</span>}
          {showPercentage && <span>{Math.round(clampedProgress)}%</span>}
        </div>
      )}
      <div className={`w-full ${sizeClasses[size]} bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden`}>
        <div
          className={`${sizeClasses[size]} ${colorClasses[color]} transition-all duration-300 ease-out rounded-full`}
          style={{ width: `${clampedProgress}%` }}
        />
      </div>
    </div>
  );
};

/**
 * Dots Loading Indicator
 */
export const DotsLoading: React.FC<{
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}> = ({ size = 'md', className = '' }) => {
  const sizes = {
    sm: 'w-1.5 h-1.5',
    md: 'w-2 h-2',
    lg: 'w-3 h-3',
  };

  return (
    <div className={`flex items-center gap-1 ${className}`}>
      <div className={`${sizes[size]} bg-primary-600 rounded-full animate-bounce`} style={{ animationDelay: '0ms' }} />
      <div className={`${sizes[size]} bg-primary-600 rounded-full animate-bounce`} style={{ animationDelay: '150ms' }} />
      <div className={`${sizes[size]} bg-primary-600 rounded-full animate-bounce`} style={{ animationDelay: '300ms' }} />
    </div>
  );
};

/**
 * Spinner Component
 */
export const Spinner: React.FC<{
  size?: 'sm' | 'md' | 'lg' | 'xl';
  color?: 'primary' | 'success' | 'warning' | 'danger' | 'gray';
  className?: string;
}> = ({ size = 'md', color = 'primary', className = '' }) => {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12',
  };

  const colors = {
    primary: 'text-primary-600',
    success: 'text-success-600',
    warning: 'text-warning-600',
    danger: 'text-danger-600',
    gray: 'text-gray-600',
  };

  return (
    <FiLoader className={`${sizes[size]} ${colors[color]} animate-spin ${className}`} />
  );
};

/**
 * Circular Progress Component
 */
export const CircularProgress: React.FC<{
  progress: number;
  size?: number;
  strokeWidth?: number;
  color?: 'primary' | 'success' | 'warning' | 'danger';
  showLabel?: boolean;
  className?: string;
}> = ({
  progress,
  size = 100,
  strokeWidth = 8,
  color = 'primary',
  showLabel = true,
  className = '',
}) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const clampedProgress = Math.max(0, Math.min(100, progress));
  const offset = circumference - (clampedProgress / 100) * circumference;

  const colorClasses = {
    primary: 'stroke-primary-600',
    success: 'stroke-success-600',
    warning: 'stroke-warning-600',
    danger: 'stroke-danger-600',
  };

  return (
    <div className={`relative inline-flex items-center justify-center ${className}`}>
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="none"
          className="text-gray-200 dark:text-gray-700"
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className={`${colorClasses[color]} transition-all duration-300 ease-out`}
        />
      </svg>
      {showLabel && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            {Math.round(clampedProgress)}%
          </span>
        </div>
      )}
    </div>
  );
};

export default LoadingOverlay;
