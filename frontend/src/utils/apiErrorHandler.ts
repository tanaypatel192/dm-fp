/**
 * API Error Handler Utility
 *
 * Provides retry logic, network detection, and user-friendly error messages
 */

import { AxiosError } from 'axios';

export interface RetryConfig {
  maxRetries: number;
  baseDelay: number;
  maxDelay: number;
  retryableStatuses: number[];
}

const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxRetries: 3,
  baseDelay: 1000, // 1 second
  maxDelay: 10000, // 10 seconds
  retryableStatuses: [408, 429, 500, 502, 503, 504],
};

/**
 * Sleep utility for retry delays
 */
const sleep = (ms: number): Promise<void> => {
  return new Promise((resolve) => setTimeout(resolve, ms));
};

/**
 * Calculate exponential backoff delay
 */
export function calculateBackoffDelay(attempt: number, config: RetryConfig): number {
  const delay = Math.min(
    config.baseDelay * Math.pow(2, attempt),
    config.maxDelay
  );
  // Add jitter (random 0-20%)
  const jitter = delay * 0.2 * Math.random();
  return delay + jitter;
}

/**
 * Check if error is retryable
 */
export function isRetryableError(error: any, config: RetryConfig): boolean {
  if (!error.response) {
    // Network errors are retryable
    return true;
  }

  return config.retryableStatuses.includes(error.response.status);
}

/**
 * Retry wrapper for async functions
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  config: Partial<RetryConfig> = {}
): Promise<T> {
  const retryConfig = { ...DEFAULT_RETRY_CONFIG, ...config };
  let lastError: any;

  for (let attempt = 0; attempt <= retryConfig.maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      // Don't retry if this is the last attempt
      if (attempt === retryConfig.maxRetries) {
        break;
      }

      // Don't retry if error is not retryable
      if (!isRetryableError(error, retryConfig)) {
        break;
      }

      // Calculate delay and wait
      const delay = calculateBackoffDelay(attempt, retryConfig);
      console.log(`[Retry] Attempt ${attempt + 1}/${retryConfig.maxRetries} failed. Retrying in ${delay}ms...`);
      await sleep(delay);
    }
  }

  throw lastError;
}

/**
 * Check if user is online
 */
export function isOnline(): boolean {
  return navigator.onLine;
}

/**
 * Network status monitoring
 */
export class NetworkMonitor {
  private listeners: Set<(online: boolean) => void> = new Set();

  constructor() {
    if (typeof window !== 'undefined') {
      window.addEventListener('online', () => this.handleStatusChange(true));
      window.addEventListener('offline', () => this.handleStatusChange(false));
    }
  }

  private handleStatusChange(online: boolean) {
    console.log(`[Network] Status changed: ${online ? 'online' : 'offline'}`);
    this.listeners.forEach((listener) => listener(online));
  }

  subscribe(listener: (online: boolean) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  isOnline(): boolean {
    return isOnline();
  }
}

// Singleton instance
export const networkMonitor = new NetworkMonitor();

/**
 * Get user-friendly error message
 */
export function getErrorMessage(error: any): string {
  // Network error
  if (!error.response && error.request) {
    if (!isOnline()) {
      return 'No internet connection. Please check your network and try again.';
    }
    return 'Unable to connect to the server. Please check your internet connection.';
  }

  // Axios error with response
  if (error.response) {
    const status = error.response.status;
    const data = error.response.data;

    // Try to get message from response data
    if (data?.detail) {
      return typeof data.detail === 'string' ? data.detail : 'An error occurred';
    }
    if (data?.message) {
      return data.message;
    }

    // Default messages by status code
    switch (status) {
      case 400:
        return 'Invalid request. Please check your input.';
      case 401:
        return 'Authentication required. Please log in.';
      case 403:
        return 'You do not have permission to perform this action.';
      case 404:
        return 'Resource not found.';
      case 408:
        return 'Request timeout. Please try again.';
      case 409:
        return 'Conflict. This resource already exists.';
      case 422:
        return 'Validation error. Please check your input.';
      case 429:
        return 'Too many requests. Please wait and try again.';
      case 500:
        return 'Internal server error. Please try again later.';
      case 502:
        return 'Bad gateway. The server is temporarily unavailable.';
      case 503:
        return 'Service unavailable. Please try again later.';
      case 504:
        return 'Gateway timeout. The server took too long to respond.';
      default:
        return `An error occurred (${status}). Please try again.`;
    }
  }

  // Request setup error
  if (error.message) {
    if (error.message.includes('timeout')) {
      return 'Request timed out. Please try again.';
    }
    if (error.message.includes('Network Error')) {
      return 'Network error. Please check your connection.';
    }
    return error.message;
  }

  return 'An unexpected error occurred. Please try again.';
}

/**
 * Get error severity level
 */
export function getErrorSeverity(error: any): 'low' | 'medium' | 'high' {
  if (!error.response) {
    return 'high'; // Network errors are high severity
  }

  const status = error.response.status;

  if (status >= 500) {
    return 'high'; // Server errors
  }
  if (status === 401 || status === 403) {
    return 'high'; // Auth errors
  }
  if (status >= 400) {
    return 'medium'; // Client errors
  }

  return 'low';
}

/**
 * Extract validation errors from response
 */
export function extractValidationErrors(error: any): Record<string, string[]> | null {
  if (!error.response?.data) {
    return null;
  }

  const data = error.response.data;

  // FastAPI validation errors
  if (data.detail && Array.isArray(data.detail)) {
    const errors: Record<string, string[]> = {};
    data.detail.forEach((err: any) => {
      if (err.loc && err.msg) {
        const field = err.loc.join('.');
        if (!errors[field]) {
          errors[field] = [];
        }
        errors[field].push(err.msg);
      }
    });
    return errors;
  }

  // Generic validation errors
  if (data.errors && typeof data.errors === 'object') {
    return data.errors;
  }

  return null;
}

/**
 * Timeout error detection
 */
export function isTimeoutError(error: any): boolean {
  return (
    error.code === 'ECONNABORTED' ||
    error.message?.includes('timeout') ||
    error.response?.status === 408
  );
}

/**
 * Network error detection
 */
export function isNetworkError(error: any): boolean {
  return (
    !error.response &&
    error.request &&
    (error.message?.includes('Network Error') || !isOnline())
  );
}

/**
 * Server error detection (5xx)
 */
export function isServerError(error: any): boolean {
  return error.response?.status >= 500;
}

/**
 * Client error detection (4xx)
 */
export function isClientError(error: any): boolean {
  const status = error.response?.status;
  return status >= 400 && status < 500;
}

/**
 * Auth error detection (401, 403)
 */
export function isAuthError(error: any): boolean {
  const status = error.response?.status;
  return status === 401 || status === 403;
}

/**
 * Rate limit error detection (429)
 */
export function isRateLimitError(error: any): boolean {
  return error.response?.status === 429;
}

/**
 * Error logger for tracking/debugging
 */
export function logError(error: any, context?: string): void {
  const timestamp = new Date().toISOString();
  const errorInfo = {
    timestamp,
    context,
    message: getErrorMessage(error),
    severity: getErrorSeverity(error),
    status: error.response?.status,
    url: error.config?.url,
    method: error.config?.method,
    data: error.response?.data,
    online: isOnline(),
  };

  console.error('[Error Log]', errorInfo);

  // Here you could send to an error tracking service
  // trackError(errorInfo);
}

/**
 * Handle API error with user feedback
 */
export function handleApiError(error: any, context?: string): {
  message: string;
  severity: 'low' | 'medium' | 'high';
  validationErrors: Record<string, string[]> | null;
  shouldRetry: boolean;
} {
  logError(error, context);

  return {
    message: getErrorMessage(error),
    severity: getErrorSeverity(error),
    validationErrors: extractValidationErrors(error),
    shouldRetry: isRetryableError(error, DEFAULT_RETRY_CONFIG),
  };
}

/**
 * Create enhanced axios error handler
 */
export function createErrorHandler(options?: {
  onAuthError?: () => void;
  onNetworkError?: () => void;
  onServerError?: () => void;
}) {
  return (error: AxiosError) => {
    if (isAuthError(error) && options?.onAuthError) {
      options.onAuthError();
    }

    if (isNetworkError(error) && options?.onNetworkError) {
      options.onNetworkError();
    }

    if (isServerError(error) && options?.onServerError) {
      options.onServerError();
    }

    return Promise.reject(error);
  };
}

export default {
  withRetry,
  getErrorMessage,
  getErrorSeverity,
  handleApiError,
  isOnline,
  networkMonitor,
  extractValidationErrors,
  isTimeoutError,
  isNetworkError,
  isServerError,
  isClientError,
  isAuthError,
  isRateLimitError,
};
