/**
 * Google Analytics 4 Configuration
 *
 * User behavior tracking and analytics setup
 */

import ReactGA from 'react-ga4';

/**
 * Initialize Google Analytics
 *
 * Call this once at app startup
 */
export const initAnalytics = () => {
  const gaId = import.meta.env.VITE_GA_MEASUREMENT_ID;
  const environment = import.meta.env.MODE || 'development';

  // Only initialize GA if measurement ID is provided
  if (!gaId) {
    console.warn('Google Analytics ID not provided. Analytics disabled.');
    return;
  }

  // Don't track in development unless explicitly enabled
  if (environment === 'development' && !import.meta.env.VITE_GA_DEBUG) {
    console.log('Google Analytics disabled in development');
    return;
  }

  ReactGA.initialize(gaId, {
    gaOptions: {
      anonymize_ip: true, // Anonymize IP addresses
      cookie_flags: 'SameSite=None;Secure', // Secure cookies
    },
    gtagOptions: {
      send_page_view: false, // We'll send page views manually
    },
  });

  console.log(`✓ Google Analytics initialized (${environment})`);
};

/**
 * Track page view
 *
 * Call this on route changes
 */
export const trackPageView = (path: string, title?: string) => {
  if (!isAnalyticsEnabled()) return;

  ReactGA.send({
    hitType: 'pageview',
    page: path,
    title: title || document.title,
  });
};

/**
 * Track custom event
 *
 * Use this to track user interactions
 */
export const trackEvent = (
  category: string,
  action: string,
  label?: string,
  value?: number
) => {
  if (!isAnalyticsEnabled()) return;

  ReactGA.event({
    category,
    action,
    label,
    value,
  });
};

/**
 * Predefined event tracking functions
 */
export const analytics = {
  // Prediction events
  prediction: {
    started: (model: string) => {
      trackEvent('Prediction', 'Started', model);
    },
    completed: (model: string, riskLevel: string) => {
      trackEvent('Prediction', 'Completed', `${model} - ${riskLevel}`);
    },
    failed: (model: string, error: string) => {
      trackEvent('Prediction', 'Failed', `${model} - ${error}`);
    },
  },

  // Batch analysis events
  batch: {
    uploaded: (fileSize: number, rowCount: number) => {
      trackEvent('Batch', 'File Uploaded', `${rowCount} rows`, fileSize);
    },
    analyzed: (model: string, rowCount: number) => {
      trackEvent('Batch', 'Analyzed', model, rowCount);
    },
    exported: (format: string, rowCount: number) => {
      trackEvent('Batch', 'Exported', format, rowCount);
    },
  },

  // Report events
  report: {
    generated: (type: 'single' | 'batch' | 'comparison') => {
      trackEvent('Report', 'Generated', type);
    },
    downloaded: (type: string) => {
      trackEvent('Report', 'Downloaded', type);
    },
    printed: (type: string) => {
      trackEvent('Report', 'Printed', type);
    },
  },

  // Model events
  model: {
    selected: (model: string) => {
      trackEvent('Model', 'Selected', model);
    },
    compared: (models: string[]) => {
      trackEvent('Model', 'Comparison', models.join(' vs '));
    },
    metricsViewed: (model: string) => {
      trackEvent('Model', 'Metrics Viewed', model);
    },
  },

  // Visualization events
  visualization: {
    viewed: (chartType: string) => {
      trackEvent('Visualization', 'Viewed', chartType);
    },
    interacted: (chartType: string, action: string) => {
      trackEvent('Visualization', 'Interaction', `${chartType} - ${action}`);
    },
    exported: (chartType: string, format: string) => {
      trackEvent('Visualization', 'Exported', `${chartType} - ${format}`);
    },
  },

  // Feature events
  feature: {
    importanceViewed: (model: string) => {
      trackEvent('Feature', 'Importance Viewed', model);
    },
    shapViewed: (model: string) => {
      trackEvent('Feature', 'SHAP Viewed', model);
    },
    distributionViewed: () => {
      trackEvent('Feature', 'Distribution Viewed');
    },
  },

  // Navigation events
  navigation: {
    menuClicked: (item: string) => {
      trackEvent('Navigation', 'Menu Clicked', item);
    },
    linkClicked: (destination: string) => {
      trackEvent('Navigation', 'Link Clicked', destination);
    },
  },

  // User interaction events
  interaction: {
    buttonClicked: (buttonName: string) => {
      trackEvent('Interaction', 'Button Clicked', buttonName);
    },
    formSubmitted: (formName: string) => {
      trackEvent('Interaction', 'Form Submitted', formName);
    },
    toggleChanged: (toggleName: string, value: boolean) => {
      trackEvent('Interaction', 'Toggle Changed', toggleName, value ? 1 : 0);
    },
  },

  // Error events
  error: {
    occurred: (errorType: string, message: string) => {
      trackEvent('Error', errorType, message);
    },
    boundary: (componentName: string) => {
      trackEvent('Error', 'Error Boundary', componentName);
    },
    apiError: (endpoint: string, statusCode: number) => {
      trackEvent('Error', 'API Error', endpoint, statusCode);
    },
  },

  // Performance events
  performance: {
    slowLoad: (component: string, duration: number) => {
      trackEvent('Performance', 'Slow Load', component, duration);
    },
    apiSlow: (endpoint: string, duration: number) => {
      trackEvent('Performance', 'Slow API', endpoint, duration);
    },
  },

  // Search events
  search: {
    performed: (query: string, results: number) => {
      trackEvent('Search', 'Performed', query, results);
    },
    resultClicked: (query: string, position: number) => {
      trackEvent('Search', 'Result Clicked', query, position);
    },
  },

  // Export events
  export: {
    csv: (dataType: string, rowCount: number) => {
      trackEvent('Export', 'CSV', dataType, rowCount);
    },
    pdf: (reportType: string) => {
      trackEvent('Export', 'PDF', reportType);
    },
    json: (dataType: string) => {
      trackEvent('Export', 'JSON', dataType);
    },
  },

  // Help events
  help: {
    tooltipViewed: (tooltipId: string) => {
      trackEvent('Help', 'Tooltip Viewed', tooltipId);
    },
    documentationClicked: (section: string) => {
      trackEvent('Help', 'Documentation Clicked', section);
    },
  },
};

/**
 * Track timing
 *
 * Measure how long an operation takes
 */
export const trackTiming = (
  category: string,
  variable: string,
  value: number,
  label?: string
) => {
  if (!isAnalyticsEnabled()) return;

  ReactGA.event({
    category: 'timing',
    action: category,
    label: `${variable}${label ? ` - ${label}` : ''}`,
    value: Math.round(value),
  });
};

/**
 * Track exception
 *
 * Track JavaScript errors
 */
export const trackException = (description: string, fatal: boolean = false) => {
  if (!isAnalyticsEnabled()) return;

  ReactGA.event({
    category: 'exception',
    action: description,
    label: fatal ? 'fatal' : 'non-fatal',
  });
};

/**
 * Set user properties
 *
 * Identify user characteristics
 */
export const setUserProperties = (properties: Record<string, any>) => {
  if (!isAnalyticsEnabled()) return;

  ReactGA.set(properties);
};

/**
 * Set user ID
 *
 * Track across sessions
 */
export const setUserId = (userId: string) => {
  if (!isAnalyticsEnabled()) return;

  ReactGA.set({ userId });
};

/**
 * Clear user ID
 */
export const clearUserId = () => {
  if (!isAnalyticsEnabled()) return;

  ReactGA.set({ userId: null });
};

/**
 * Check if analytics is enabled
 */
export const isAnalyticsEnabled = (): boolean => {
  const gaId = import.meta.env.VITE_GA_MEASUREMENT_ID;
  const environment = import.meta.env.MODE || 'development';

  if (!gaId) return false;
  if (environment === 'development' && !import.meta.env.VITE_GA_DEBUG) return false;

  return true;
};

/**
 * Performance timing utility
 *
 * Measure and track component/operation performance
 */
export class PerformanceTracker {
  private startTime: number;
  private category: string;
  private variable: string;
  private label?: string;

  constructor(category: string, variable: string, label?: string) {
    this.category = category;
    this.variable = variable;
    this.label = label;
    this.startTime = performance.now();
  }

  end() {
    const duration = performance.now() - this.startTime;
    trackTiming(this.category, this.variable, duration, this.label);
    return duration;
  }
}

/**
 * HOC to track component mount/unmount performance
 */
export const withPerformanceTracking = <P extends object>(
  Component: React.ComponentType<P>,
  componentName: string
) => {
  return (props: P) => {
    const tracker = new PerformanceTracker('Component', 'Mount', componentName);

    React.useEffect(() => {
      const duration = tracker.end();

      // Track slow loads (> 1 second)
      if (duration > 1000) {
        analytics.performance.slowLoad(componentName, Math.round(duration));
      }

      return () => {
        trackEvent('Component', 'Unmount', componentName);
      };
    }, []);

    return <Component {...props} />;
  };
};
