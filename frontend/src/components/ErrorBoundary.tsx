import React, { Component, ErrorInfo, ReactNode } from 'react';
import { FiAlertTriangle, FiHome, FiRefreshCw, FiChevronDown, FiChevronUp } from 'react-icons/fi';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  showDetails: boolean;
}

/**
 * ErrorBoundary Component
 *
 * Catches React errors in child components and displays user-friendly error UI
 * Provides options to retry or navigate home
 */
class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log error to console
    console.error('ErrorBoundary caught an error:', error, errorInfo);

    // Update state with error details
    this.setState({
      error,
      errorInfo,
    });

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Here you could also log to an error tracking service like Sentry
    // logErrorToService(error, errorInfo);
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false,
    });
  };

  handleGoHome = () => {
    window.location.href = '/';
  };

  toggleDetails = () => {
    this.setState((prevState) => ({
      showDetails: !prevState.showDetails,
    }));
  };

  getErrorType(error: Error): { title: string; description: string; color: string } {
    const message = error.message.toLowerCase();

    if (message.includes('network') || message.includes('fetch')) {
      return {
        title: 'Network Error',
        description: 'Unable to connect to the server. Please check your internet connection and try again.',
        color: 'text-warning-600',
      };
    }

    if (message.includes('timeout')) {
      return {
        title: 'Request Timeout',
        description: 'The request took too long to complete. Please try again.',
        color: 'text-warning-600',
      };
    }

    if (message.includes('permission') || message.includes('unauthorized')) {
      return {
        title: 'Permission Denied',
        description: 'You do not have permission to access this resource.',
        color: 'text-danger-600',
      };
    }

    if (message.includes('not found') || message.includes('404')) {
      return {
        title: 'Not Found',
        description: 'The requested resource could not be found.',
        color: 'text-warning-600',
      };
    }

    // Default error
    return {
      title: 'Unexpected Error',
      description: 'Something went wrong. Please try again or contact support if the problem persists.',
      color: 'text-danger-600',
    };
  }

  render() {
    if (this.state.hasError) {
      // Custom fallback UI provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const { error, errorInfo, showDetails } = this.state;
      const errorType = error ? this.getErrorType(error) : null;

      // Default error UI
      return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center p-4">
          <div className="max-w-2xl w-full">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-8 border border-gray-200 dark:border-gray-700">
              {/* Error Icon */}
              <div className="flex justify-center mb-6">
                <div className="w-20 h-20 bg-danger-100 dark:bg-danger-900/30 rounded-full flex items-center justify-center">
                  <FiAlertTriangle className={`w-10 h-10 ${errorType?.color || 'text-danger-600'}`} />
                </div>
              </div>

              {/* Error Title */}
              <h1 className="text-3xl font-bold text-center text-gray-900 dark:text-gray-100 mb-4">
                {errorType?.title || 'Oops! Something went wrong'}
              </h1>

              {/* Error Description */}
              <p className="text-center text-gray-600 dark:text-gray-400 mb-8">
                {errorType?.description || 'We encountered an unexpected error. Please try refreshing the page.'}
              </p>

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row gap-4 justify-center mb-6">
                <button
                  onClick={this.handleReset}
                  className="flex items-center justify-center gap-2 px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors font-medium"
                >
                  <FiRefreshCw className="w-5 h-5" />
                  Try Again
                </button>

                <button
                  onClick={this.handleGoHome}
                  className="flex items-center justify-center gap-2 px-6 py-3 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors font-medium"
                >
                  <FiHome className="w-5 h-5" />
                  Go Home
                </button>
              </div>

              {/* Error Details Toggle */}
              {error && (
                <div className="mt-6">
                  <button
                    onClick={this.toggleDetails}
                    className="w-full flex items-center justify-center gap-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 transition-colors"
                  >
                    {showDetails ? (
                      <>
                        <FiChevronUp className="w-4 h-4" />
                        Hide Technical Details
                      </>
                    ) : (
                      <>
                        <FiChevronDown className="w-4 h-4" />
                        Show Technical Details
                      </>
                    )}
                  </button>

                  {showDetails && (
                    <div className="mt-4 p-4 bg-gray-100 dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700">
                      <div className="space-y-4">
                        {/* Error Message */}
                        <div>
                          <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                            Error Message:
                          </h3>
                          <pre className="text-xs text-danger-600 dark:text-danger-400 bg-white dark:bg-gray-800 p-3 rounded border border-gray-200 dark:border-gray-700 overflow-x-auto">
                            {error.toString()}
                          </pre>
                        </div>

                        {/* Component Stack */}
                        {errorInfo && errorInfo.componentStack && (
                          <div>
                            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                              Component Stack:
                            </h3>
                            <pre className="text-xs text-gray-600 dark:text-gray-400 bg-white dark:bg-gray-800 p-3 rounded border border-gray-200 dark:border-gray-700 overflow-x-auto max-h-48 overflow-y-auto">
                              {errorInfo.componentStack}
                            </pre>
                          </div>
                        )}

                        {/* Stack Trace */}
                        {error.stack && (
                          <div>
                            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                              Stack Trace:
                            </h3>
                            <pre className="text-xs text-gray-600 dark:text-gray-400 bg-white dark:bg-gray-800 p-3 rounded border border-gray-200 dark:border-gray-700 overflow-x-auto max-h-48 overflow-y-auto">
                              {error.stack}
                            </pre>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Help Text */}
              <div className="mt-8 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
                <p className="text-sm text-blue-800 dark:text-blue-300">
                  <strong>Need help?</strong> If this error persists, please contact support with the error
                  details above. We're here to help!
                </p>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
