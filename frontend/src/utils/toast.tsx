/**
 * Toast Notification Utility
 *
 * Wraps react-toastify with custom configurations and action buttons
 */

import { toast as reactToast, ToastOptions, ToastContent, Id } from 'react-toastify';
import { FiCheckCircle, FiXCircle, FiAlertTriangle, FiInfo, FiX } from 'react-icons/fi';
import React from 'react';

// Default toast configuration
const defaultConfig: ToastOptions = {
  position: 'top-right',
  autoClose: 5000,
  hideProgressBar: false,
  closeOnClick: true,
  pauseOnHover: true,
  draggable: true,
  progress: undefined,
  theme: 'colored',
};

/**
 * Custom toast component with icon
 */
const ToastMessage: React.FC<{
  icon: React.ReactNode;
  title?: string;
  message: string;
}> = ({ icon, title, message }) => (
  <div className="flex items-start gap-3">
    <div className="flex-shrink-0 mt-0.5">{icon}</div>
    <div className="flex-1 min-w-0">
      {title && <div className="font-semibold mb-1">{title}</div>}
      <div className="text-sm">{message}</div>
    </div>
  </div>
);

/**
 * Toast with action buttons
 */
const ToastWithActions: React.FC<{
  icon: React.ReactNode;
  message: string;
  actions?: Array<{
    label: string;
    onClick: () => void;
  }>;
}> = ({ icon, message, actions }) => (
  <div>
    <div className="flex items-start gap-3 mb-3">
      <div className="flex-shrink-0 mt-0.5">{icon}</div>
      <div className="flex-1 min-w-0">
        <div className="text-sm">{message}</div>
      </div>
    </div>
    {actions && actions.length > 0 && (
      <div className="flex gap-2 mt-2">
        {actions.map((action, index) => (
          <button
            key={index}
            onClick={() => {
              action.onClick();
              reactToast.dismiss();
            }}
            className="px-3 py-1 text-xs font-medium bg-white/20 hover:bg-white/30 rounded transition-colors"
          >
            {action.label}
          </button>
        ))}
      </div>
    )}
  </div>
);

/**
 * Toast notification utilities
 */
export const toast = {
  /**
   * Success toast (green)
   */
  success: (message: string, options?: ToastOptions): Id => {
    return reactToast.success(
      <ToastMessage icon={<FiCheckCircle className="w-5 h-5" />} message={message} />,
      {
        ...defaultConfig,
        ...options,
      }
    );
  },

  /**
   * Error toast (red)
   */
  error: (message: string, options?: ToastOptions): Id => {
    return reactToast.error(
      <ToastMessage icon={<FiXCircle className="w-5 h-5" />} message={message} />,
      {
        ...defaultConfig,
        autoClose: 7000, // Errors stay longer
        ...options,
      }
    );
  },

  /**
   * Warning toast (yellow)
   */
  warning: (message: string, options?: ToastOptions): Id => {
    return reactToast.warning(
      <ToastMessage icon={<FiAlertTriangle className="w-5 h-5" />} message={message} />,
      {
        ...defaultConfig,
        ...options,
      }
    );
  },

  /**
   * Info toast (blue)
   */
  info: (message: string, options?: ToastOptions): Id => {
    return reactToast.info(
      <ToastMessage icon={<FiInfo className="w-5 h-5" />} message={message} />,
      {
        ...defaultConfig,
        ...options,
      }
    );
  },

  /**
   * Success toast with title
   */
  successWithTitle: (title: string, message: string, options?: ToastOptions): Id => {
    return reactToast.success(
      <ToastMessage icon={<FiCheckCircle className="w-5 h-5" />} title={title} message={message} />,
      {
        ...defaultConfig,
        ...options,
      }
    );
  },

  /**
   * Error toast with title
   */
  errorWithTitle: (title: string, message: string, options?: ToastOptions): Id => {
    return reactToast.error(
      <ToastMessage icon={<FiXCircle className="w-5 h-5" />} title={title} message={message} />,
      {
        ...defaultConfig,
        autoClose: 7000,
        ...options,
      }
    );
  },

  /**
   * Toast with action buttons (e.g., Undo, Retry)
   */
  withActions: (
    message: string,
    actions: Array<{ label: string; onClick: () => void }>,
    type: 'success' | 'error' | 'warning' | 'info' = 'info',
    options?: ToastOptions
  ): Id => {
    const icons = {
      success: <FiCheckCircle className="w-5 h-5" />,
      error: <FiXCircle className="w-5 h-5" />,
      warning: <FiAlertTriangle className="w-5 h-5" />,
      info: <FiInfo className="w-5 h-5" />,
    };

    const toastFn = reactToast[type];
    return toastFn(<ToastWithActions icon={icons[type]} message={message} actions={actions} />, {
      ...defaultConfig,
      ...options,
    });
  },

  /**
   * Promise toast - shows loading, success, or error based on promise
   */
  promise: <T,>(
    promise: Promise<T>,
    messages: {
      pending: string;
      success: string | ((data: T) => string);
      error: string | ((error: any) => string);
    },
    options?: ToastOptions
  ): Promise<T> => {
    return reactToast.promise(
      promise,
      {
        pending: {
          render: messages.pending,
          icon: '⏳',
        },
        success: {
          render: typeof messages.success === 'function'
            ? ({ data }) => messages.success(data as T)
            : messages.success,
          icon: <FiCheckCircle className="w-5 h-5" />,
        },
        error: {
          render: typeof messages.error === 'function'
            ? ({ data }) => messages.error(data)
            : messages.error,
          icon: <FiXCircle className="w-5 h-5" />,
        },
      },
      {
        ...defaultConfig,
        ...options,
      }
    );
  },

  /**
   * Loading toast that must be manually dismissed
   */
  loading: (message: string, options?: ToastOptions): Id => {
    return reactToast.info(message, {
      ...defaultConfig,
      autoClose: false,
      closeButton: false,
      draggable: false,
      ...options,
    });
  },

  /**
   * Update an existing toast
   */
  update: (
    toastId: Id,
    options: {
      type?: 'success' | 'error' | 'warning' | 'info';
      render?: ToastContent;
      autoClose?: number | false;
    }
  ): void => {
    reactToast.update(toastId, options);
  },

  /**
   * Dismiss a specific toast
   */
  dismiss: (toastId?: Id): void => {
    reactToast.dismiss(toastId);
  },

  /**
   * Dismiss all toasts
   */
  dismissAll: (): void => {
    reactToast.dismiss();
  },

  /**
   * Check if a toast is active
   */
  isActive: (toastId: Id): boolean => {
    return reactToast.isActive(toastId);
  },
};

/**
 * Toast presets for common scenarios
 */
export const toastPresets = {
  /**
   * API success
   */
  apiSuccess: (action: string = 'Operation'): Id => {
    return toast.success(`${action} completed successfully!`);
  },

  /**
   * API error
   */
  apiError: (error?: string): Id => {
    return toast.error(error || 'An error occurred. Please try again.');
  },

  /**
   * Network error
   */
  networkError: (): Id => {
    return toast.error('Network error. Please check your connection and try again.');
  },

  /**
   * Validation error
   */
  validationError: (message?: string): Id => {
    return toast.warning(message || 'Please check your input and try again.');
  },

  /**
   * Save success
   */
  saveSuccess: (): Id => {
    return toast.success('Changes saved successfully!');
  },

  /**
   * Delete success
   */
  deleteSuccess: (): Id => {
    return toast.success('Deleted successfully!');
  },

  /**
   * Copy success
   */
  copySuccess: (): Id => {
    return toast.success('Copied to clipboard!');
  },

  /**
   * Upload success
   */
  uploadSuccess: (): Id => {
    return toast.success('File uploaded successfully!');
  },

  /**
   * Download started
   */
  downloadStarted: (): Id => {
    return toast.info('Download started...');
  },

  /**
   * Coming soon
   */
  comingSoon: (): Id => {
    return toast.info('This feature is coming soon!');
  },

  /**
   * Prediction success
   */
  predictionSuccess: (): Id => {
    return toast.success('Prediction completed successfully!');
  },

  /**
   * Batch processing complete
   */
  batchComplete: (count: number): Id => {
    return toast.successWithTitle(
      'Batch Processing Complete',
      `Successfully processed ${count} patient${count !== 1 ? 's' : ''}.`
    );
  },

  /**
   * Session expired
   */
  sessionExpired: (): Id => {
    return toast.warning('Your session has expired. Please log in again.');
  },
};

/**
 * Export default react-toastify functions for advanced usage
 */
export { reactToast };
export type { ToastOptions, Id as ToastId };
