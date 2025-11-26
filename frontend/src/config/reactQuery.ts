/**
 * React Query Configuration
 *
 * Centralized configuration for @tanstack/react-query
 * with optimal caching strategies and error handling
 */

import { QueryClient, DefaultOptions } from '@tanstack/react-query';

/**
 * Default options for React Query
 *
 * These options are applied to all queries and mutations
 * unless overridden at the component level
 */
const queryConfig: DefaultOptions = {
  queries: {
    // Stale time: How long data is considered fresh
    // 5 minutes for most queries
    staleTime: 5 * 60 * 1000,

    // Cache time: How long unused data stays in cache
    // 10 minutes before garbage collection
    gcTime: 10 * 60 * 1000,

    // Retry configuration
    retry: (failureCount: number, error: any) => {
      // Don't retry on 4xx errors (client errors)
      if (error?.response?.status >= 400 && error?.response?.status < 500) {
        return false;
      }
      // Retry up to 3 times for server errors
      return failureCount < 3;
    },

    // Retry delay with exponential backoff
    retryDelay: (attemptIndex: number) => {
      return Math.min(1000 * 2 ** attemptIndex, 30000);
    },

    // Refetch configuration
    refetchOnWindowFocus: false, // Don't refetch when window regains focus
    refetchOnReconnect: true, // Refetch when network reconnects
    refetchOnMount: true, // Refetch when component mounts

    // Network mode
    networkMode: 'online', // Only fetch when online
  },

  mutations: {
    // Retry mutations once on failure
    retry: 1,

    // Network mode
    networkMode: 'online',
  },
};

/**
 * Create and configure QueryClient instance
 */
export const queryClient = new QueryClient({
  defaultOptions: queryConfig,
});

/**
 * Query Keys Factory
 *
 * Centralized query key management for consistency
 * and type safety
 */
export const queryKeys = {
  // Health check
  health: ['health'] as const,

  // Predictions
  predictions: {
    all: ['predictions'] as const,
    single: (id: string) => ['predictions', id] as const,
    byModel: (model: string) => ['predictions', 'model', model] as const,
  },

  // Models
  models: {
    all: ['models'] as const,
    metrics: (model: string) => ['models', model, 'metrics'] as const,
    info: (model: string) => ['models', model, 'info'] as const,
    comparison: (models: string[]) => ['models', 'comparison', ...models] as const,
  },

  // Features
  features: {
    importance: (model: string) => ['features', model, 'importance'] as const,
    distributions: ['features', 'distributions'] as const,
  },

  // Batch operations
  batch: {
    all: ['batch'] as const,
    job: (id: string) => ['batch', id] as const,
    status: (id: string) => ['batch', id, 'status'] as const,
  },

  // Performance metrics
  performance: {
    all: ['performance'] as const,
    api: ['performance', 'api'] as const,
    models: ['performance', 'models'] as const,
    system: ['performance', 'system'] as const,
  },
};

/**
 * Cache Time Presets
 *
 * Common cache time configurations for different data types
 */
export const cachePresets = {
  // Very fast changing data (real-time metrics)
  realtime: {
    staleTime: 0,
    gcTime: 1 * 60 * 1000, // 1 minute
  },

  // Fast changing data (API metrics)
  fast: {
    staleTime: 1 * 60 * 1000, // 1 minute
    gcTime: 5 * 60 * 1000, // 5 minutes
  },

  // Normal changing data (predictions, model info)
  normal: {
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
  },

  // Slow changing data (model metrics, feature importance)
  slow: {
    staleTime: 15 * 60 * 1000, // 15 minutes
    gcTime: 30 * 60 * 1000, // 30 minutes
  },

  // Static data (model names, feature lists)
  static: {
    staleTime: 60 * 60 * 1000, // 1 hour
    gcTime: 24 * 60 * 60 * 1000, // 24 hours
  },

  // Infinite cache (won't be automatically refetched)
  infinite: {
    staleTime: Infinity,
    gcTime: Infinity,
  },
};

/**
 * Prefetch utility for warming up cache
 *
 * Use this to prefetch data before it's needed
 */
export const prefetchQuery = async <T>(
  queryKey: readonly unknown[],
  queryFn: () => Promise<T>,
  options?: {
    staleTime?: number;
    gcTime?: number;
  }
) => {
  await queryClient.prefetchQuery({
    queryKey,
    queryFn,
    staleTime: options?.staleTime,
    gcTime: options?.gcTime,
  });
};

/**
 * Invalidate queries utility
 *
 * Use this to force refetch of specific queries
 */
export const invalidateQueries = async (queryKey: readonly unknown[]) => {
  await queryClient.invalidateQueries({ queryKey });
};

/**
 * Clear all cache
 *
 * Nuclear option - clears all cached data
 */
export const clearCache = () => {
  queryClient.clear();
};

/**
 * Get cached data
 *
 * Retrieve cached data without triggering a fetch
 */
export const getCachedData = <T>(queryKey: readonly unknown[]): T | undefined => {
  return queryClient.getQueryData<T>(queryKey);
};

/**
 * Set cached data
 *
 * Manually set data in cache
 */
export const setCachedData = <T>(queryKey: readonly unknown[], data: T) => {
  queryClient.setQueryData<T>(queryKey, data);
};

/**
 * React Query DevTools configuration
 */
export const devToolsConfig = {
  initialIsOpen: false,
  position: 'bottom-right' as const,
  buttonPosition: 'bottom-right' as const,
};
