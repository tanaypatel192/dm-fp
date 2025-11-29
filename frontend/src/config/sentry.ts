/**
 * Sentry Configuration
 *
 * Error tracking and performance monitoring setup
 */

import * as Sentry from '@sentry/react';

/**
 * Initialize Sentry
 *
 * Call this once at app startup
 */
export const initSentry = () => {
  // Get environment from env variables
  const environment = import.meta.env.MODE || 'development';
  const dsn = import.meta.env.VITE_SENTRY_DSN;

  // Only initialize Sentry if DSN is provided
  if (!dsn) {
    console.warn('Sentry DSN not provided. Error tracking disabled.');
    return;
  }

  Sentry.init({
    dsn,
    environment,

    // Integrations
    integrations: [
      // Browser tracing for performance monitoring
      Sentry.browserTracingIntegration({
        // Trace navigation and interactions
        tracePropagationTargets: [
          'localhost',
          /^https:\/\/.*\.yourdomain\.com/,
          /^\/api\//,
        ],
      }),

      // Replay integration for session replay
      Sentry.replayIntegration({
        // Mask all text and user input by default
        maskAllText: true,
        maskAllInputs: true,
        blockAllMedia: true,
      }),

      // Capture console errors
      Sentry.captureConsoleIntegration({
        levels: ['error'],
      }),

      // HTTP client errors
      Sentry.httpClientIntegration(),
    ],

    // Performance monitoring
    tracesSampleRate: environment === 'production' ? 0.1 : 1.0, // 10% in prod, 100% in dev

    // Session replay
    replaysSessionSampleRate: environment === 'production' ? 0.1 : 0.5, // 10% in prod
    replaysOnErrorSampleRate: 1.0, // Always capture replays on errors

    // Release tracking
    release: import.meta.env.VITE_APP_VERSION || 'unknown',

    // Before send hook - filter sensitive data
    beforeSend(event, hint) {
      // Don't send events in development (optional)
      if (environment === 'development' && !import.meta.env.VITE_SENTRY_DEBUG) {
        console.error('Sentry event (not sent in dev):', event, hint);
        return null;
      }

      // Filter out sensitive data from event
      if (event.request) {
        // Remove sensitive headers
        delete event.request.headers?.['Authorization'];
        delete event.request.headers?.['X-API-Key'];

        // Remove sensitive query parameters
        if (event.request.query_string) {
          event.request.query_string = event.request.query_string
            .replace(/api_key=[^&]*/g, 'api_key=[REDACTED]')
            .replace(/token=[^&]*/g, 'token=[REDACTED]');
        }
      }

      // Filter sensitive data from breadcrumbs
      if (event.breadcrumbs) {
        event.breadcrumbs = event.breadcrumbs.map((breadcrumb) => {
          if (breadcrumb.data) {
            // Remove sensitive fields
            const { password, api_key, token, ...safeData } = breadcrumb.data;
            breadcrumb.data = safeData;
          }
          return breadcrumb;
        });
      }

      return event;
    },

    // Ignore certain errors
    ignoreErrors: [
      // Browser extensions
      'top.GLOBALS',
      // Random plugins/extensions
      'originalCreateNotification',
      'canvas.contentDocument',
      'MyApp_RemoveAllHighlights',
      // Network errors that are expected
      'Network request failed',
      'NetworkError',
      'Failed to fetch',
      // User cancelled requests
      'AbortError',
      'Request aborted',
      // Common non-actionable errors
      'ResizeObserver loop limit exceeded',
      'Non-Error promise rejection captured',
    ],

    // Deny URLs - don't capture errors from these sources
    denyUrls: [
      // Chrome extensions
      /extensions\//i,
      /^chrome:\/\//i,
      /^chrome-extension:\/\//i,
      // Firefox extensions
      /^moz-extension:\/\//i,
      // Other
      /^safari-extension:\/\//i,
    ],
  });

  console.log(`✓ Sentry initialized (${environment})`);
};

/**
 * Set user context for Sentry
 *
 * Call this when user logs in or when user info is available
 */
export const setSentryUser = (user: {
  id?: string;
  email?: string;
  username?: string;
  ip_address?: string;
}) => {
  Sentry.setUser(user);
};

/**
 * Clear user context
 *
 * Call this when user logs out
 */
export const clearSentryUser = () => {
  Sentry.setUser(null);
};

/**
 * Set custom context
 *
 * Add additional context to error reports
 */
export const setSentryContext = (name: string, context: Record<string, any>) => {
  Sentry.setContext(name, context);
};

/**
 * Add breadcrumb
 *
 * Add custom breadcrumb to track user actions
 */
export const addSentryBreadcrumb = (breadcrumb: {
  message: string;
  level?: 'fatal' | 'error' | 'warning' | 'log' | 'info' | 'debug';
  category?: string;
  data?: Record<string, any>;
}) => {
  Sentry.addBreadcrumb(breadcrumb);
};

/**
 * Capture exception manually
 *
 * Use this to manually report errors to Sentry
 */
export const captureException = (
  error: Error,
  context?: {
    level?: 'fatal' | 'error' | 'warning' | 'info' | 'debug';
    tags?: Record<string, string>;
    extra?: Record<string, any>;
  }
) => {
  if (context?.level) {
    Sentry.captureException(error, { level: context.level });
  } else {
    Sentry.captureException(error);
  }

  if (context?.tags) {
    Sentry.setTags(context.tags);
  }

  if (context?.extra) {
    Sentry.setExtras(context.extra);
  }
};

/**
 * Capture message manually
 *
 * Use this to report non-error messages to Sentry
 */
export const captureMessage = (
  message: string,
  level: 'fatal' | 'error' | 'warning' | 'info' | 'debug' = 'info'
) => {
  Sentry.captureMessage(message, level);
};

/**
 * Start a transaction for performance monitoring
 *
 * Use this to measure performance of specific operations
 */
export const startTransaction = (
  name: string,
  op: string,
  description?: string
) => {
  return Sentry.startTransaction({
    name,
    op,
    description,
  });
};

/**
 * Wrap component with Sentry error boundary
 *
 * Use this as a HOC or wrap your root component
 */
export const SentryErrorBoundary = Sentry.ErrorBoundary;

/**
 * Create Sentry-aware Router
 *
 * This wraps react-router-dom to automatically track navigation
 */
export const withSentryRouting = (Component: React.ComponentType<any>) => {
  return Sentry.withProfiler(Component);
};

/**
 * Create Sentry profiler component
 *
 * Use this to measure component render performance
 */
export const SentryProfiler = Sentry.Profiler;

/**
 * Environment check utilities
 */
export const sentryUtils = {
  isEnabled: () => !!import.meta.env.VITE_SENTRY_DSN,
  isProduction: () => import.meta.env.MODE === 'production',
  isDevelopment: () => import.meta.env.MODE === 'development',
};
