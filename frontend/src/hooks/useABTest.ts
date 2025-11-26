/**
 * A/B Testing React Hook
 *
 * Provides easy integration with A/B testing system
 */

import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { analytics } from '@/config/analytics';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ==================== Types ====================

export interface Variant {
  id: string;
  name: string;
  model_name: string;
  traffic_percentage: number;
  variant_type: 'control' | 'treatment';
  description: string;
}

export interface VariantMetrics {
  variant_id: string;
  total_users: number;
  total_predictions: number;
  avg_prediction_time_ms: number;
  avg_confidence: number;
  positive_predictions: number;
  negative_predictions: number;
  low_risk_count: number;
  medium_risk_count: number;
  high_risk_count: number;
  avg_interactions: number;
  conversion_rate: number;
  avg_rating: number;
  rating_count: number;
  error_count: number;
  error_rate: number;
}

export interface Experiment {
  id: string;
  name: string;
  description: string;
  status: 'draft' | 'running' | 'paused' | 'completed' | 'cancelled';
  variants: Variant[];
  created_at: string;
  started_at?: string;
  ended_at?: string;
  min_sample_size: number;
  confidence_level: number;
  target_metric: string;
  metrics: Record<string, VariantMetrics>;
}

export interface VariantAssignment {
  experiment_id: string;
  variant_id: string;
  model_name: string;
  user_id: string;
}

export interface StatisticalComparison {
  significant: boolean;
  p_value: number;
  t_statistic: number;
  confidence_level: number;
  control_mean: number;
  treatment_mean: number;
  mean_difference: number;
  relative_lift_percent: number;
  confidence_interval: {
    lower: number;
    upper: number;
  };
  control_sample_size: number;
  treatment_sample_size: number;
  recommendation: string;
  variant_name?: string;
  variant_id?: string;
}

export interface ExperimentResults {
  experiment: Experiment;
  duration_hours: number;
  comparisons: StatisticalComparison[];
  winner: {
    winner: string | null;
    variant_name?: string;
    lift?: number;
    confidence?: number;
    reason?: string;
  };
}

// ==================== API Functions ====================

const api = {
  // Get user's assigned variant
  getUserVariant: async (experimentId: string): Promise<VariantAssignment | null> => {
    const response = await fetch(`${API_BASE_URL}/api/ab-testing/experiments/${experimentId}/variant`, {
      credentials: 'include',
      headers: {
        'X-User-ID': localStorage.getItem('user_id') || '',
      },
    });

    if (!response.ok) {
      throw new Error('Failed to get user variant');
    }

    const data = await response.json();
    return data || null;
  },

  // Assign user to variant
  assignUser: async (experimentId: string): Promise<VariantAssignment> => {
    const response = await fetch(`${API_BASE_URL}/api/ab-testing/experiments/${experimentId}/assign`, {
      credentials: 'include',
      headers: {
        'X-User-ID': localStorage.getItem('user_id') || '',
      },
    });

    if (!response.ok) {
      throw new Error('Failed to assign user to variant');
    }

    return response.json();
  },

  // Track prediction
  trackPrediction: async (data: {
    experiment_id: string;
    variant_id: string;
    user_id: string;
    prediction_time_ms: number;
    confidence: number;
    prediction: number;
    risk_level: string;
    error?: boolean;
  }) => {
    const response = await fetch(`${API_BASE_URL}/api/ab-testing/track/prediction`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error('Failed to track prediction');
    }
  },

  // Track conversion
  trackConversion: async (data: {
    experiment_id: string;
    variant_id: string;
    user_id: string;
    converted: boolean;
  }) => {
    const response = await fetch(`${API_BASE_URL}/api/ab-testing/track/conversion`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error('Failed to track conversion');
    }
  },

  // Track rating
  trackRating: async (data: {
    experiment_id: string;
    variant_id: string;
    user_id: string;
    rating: number;
  }) => {
    const response = await fetch(`${API_BASE_URL}/api/ab-testing/track/rating`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error('Failed to track rating');
    }
  },

  // Track interaction
  trackInteraction: async (data: {
    experiment_id: string;
    variant_id: string;
    user_id: string;
  }) => {
    const response = await fetch(`${API_BASE_URL}/api/ab-testing/track/interaction`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error('Failed to track interaction');
    }
  },

  // Get experiment
  getExperiment: async (experimentId: string): Promise<Experiment> => {
    const response = await fetch(`${API_BASE_URL}/api/ab-testing/experiments/${experimentId}`);

    if (!response.ok) {
      throw new Error('Failed to get experiment');
    }

    return response.json();
  },

  // Get experiment results
  getExperimentResults: async (experimentId: string): Promise<ExperimentResults> => {
    const response = await fetch(`${API_BASE_URL}/api/ab-testing/experiments/${experimentId}/results`);

    if (!response.ok) {
      throw new Error('Failed to get experiment results');
    }

    return response.json();
  },

  // List experiments
  listExperiments: async (status?: string): Promise<Experiment[]> => {
    const url = new URL(`${API_BASE_URL}/api/ab-testing/experiments`);
    if (status) {
      url.searchParams.append('status', status);
    }

    const response = await fetch(url.toString());

    if (!response.ok) {
      throw new Error('Failed to list experiments');
    }

    return response.json();
  },
};

// ==================== Hooks ====================

/**
 * Use A/B Test Hook
 *
 * Main hook for integrating A/B testing into components
 *
 * @param experimentId - ID of the experiment to participate in
 * @param options - Configuration options
 * @returns A/B test state and tracking functions
 */
export const useABTest = (
  experimentId: string | null,
  options: {
    enabled?: boolean;
    autoAssign?: boolean;
  } = {}
) => {
  const { enabled = true, autoAssign = true } = options;

  const [assignment, setAssignment] = useState<VariantAssignment | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Generate or get user ID
  const getUserId = useCallback(() => {
    let userId = localStorage.getItem('user_id');
    if (!userId) {
      userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      localStorage.setItem('user_id', userId);
    }
    return userId;
  }, []);

  // Get user's assigned variant
  const { data: existingAssignment, isLoading: isCheckingAssignment } = useQuery({
    queryKey: ['ab-test', experimentId, 'assignment'],
    queryFn: () => api.getUserVariant(experimentId!),
    enabled: enabled && experimentId !== null,
    staleTime: Infinity, // Assignment doesn't change
  });

  // Assign user mutation
  const assignMutation = useMutation({
    mutationFn: () => api.assignUser(experimentId!),
    onSuccess: (data) => {
      setAssignment(data);
      analytics.model.selected(data.model_name);
    },
  });

  // Auto-assign user if not already assigned
  useEffect(() => {
    if (!enabled || !experimentId || !autoAssign) {
      setIsLoading(false);
      return;
    }

    if (isCheckingAssignment) {
      return;
    }

    if (existingAssignment) {
      setAssignment(existingAssignment);
      setIsLoading(false);
    } else if (!assignMutation.isPending) {
      assignMutation.mutate();
      setIsLoading(false);
    }
  }, [enabled, experimentId, autoAssign, existingAssignment, isCheckingAssignment]);

  // Tracking mutations
  const trackPredictionMutation = useMutation({
    mutationFn: api.trackPrediction,
  });

  const trackConversionMutation = useMutation({
    mutationFn: api.trackConversion,
  });

  const trackRatingMutation = useMutation({
    mutationFn: api.trackRating,
  });

  const trackInteractionMutation = useMutation({
    mutationFn: api.trackInteraction,
  });

  // Tracking functions
  const trackPrediction = useCallback(
    (data: {
      prediction_time_ms: number;
      confidence: number;
      prediction: number;
      risk_level: string;
      error?: boolean;
    }) => {
      if (!assignment) return;

      trackPredictionMutation.mutate({
        experiment_id: assignment.experiment_id,
        variant_id: assignment.variant_id,
        user_id: getUserId(),
        ...data,
      });
    },
    [assignment, trackPredictionMutation]
  );

  const trackConversion = useCallback(
    (converted: boolean) => {
      if (!assignment) return;

      trackConversionMutation.mutate({
        experiment_id: assignment.experiment_id,
        variant_id: assignment.variant_id,
        user_id: getUserId(),
        converted,
      });
    },
    [assignment, trackConversionMutation]
  );

  const trackRating = useCallback(
    (rating: number) => {
      if (!assignment) return;

      trackRatingMutation.mutate({
        experiment_id: assignment.experiment_id,
        variant_id: assignment.variant_id,
        user_id: getUserId(),
        rating,
      });
    },
    [assignment, trackRatingMutation]
  );

  const trackInteraction = useCallback(() => {
    if (!assignment) return;

    trackInteractionMutation.mutate({
      experiment_id: assignment.experiment_id,
      variant_id: assignment.variant_id,
      user_id: getUserId(),
    });
  }, [assignment, trackInteractionMutation]);

  return {
    // State
    isLoading,
    assignment,
    isAssigned: assignment !== null,
    variantId: assignment?.variant_id || null,
    modelName: assignment?.model_name || null,

    // Tracking
    trackPrediction,
    trackConversion,
    trackRating,
    trackInteraction,

    // Utilities
    getUserId,
  };
};

/**
 * Use Experiment Hook
 *
 * Hook for fetching experiment details
 */
export const useExperiment = (experimentId: string | null) => {
  return useQuery({
    queryKey: ['ab-test', experimentId],
    queryFn: () => api.getExperiment(experimentId!),
    enabled: experimentId !== null,
  });
};

/**
 * Use Experiment Results Hook
 *
 * Hook for fetching experiment results and statistical analysis
 */
export const useExperimentResults = (experimentId: string | null) => {
  return useQuery({
    queryKey: ['ab-test', experimentId, 'results'],
    queryFn: () => api.getExperimentResults(experimentId!),
    enabled: experimentId !== null,
  });
};

/**
 * Use Experiments List Hook
 *
 * Hook for listing all experiments
 */
export const useExperiments = (status?: string) => {
  return useQuery({
    queryKey: ['ab-test', 'experiments', status],
    queryFn: () => api.listExperiments(status),
  });
};

/**
 * Higher-Order Component for A/B Testing
 *
 * Wraps a component and automatically handles A/B test assignment
 */
export function withABTest<P extends object>(
  Component: React.ComponentType<P>,
  experimentId: string
) {
  return function ABTestWrapper(props: P) {
    const abTest = useABTest(experimentId);

    if (abTest.isLoading) {
      return <div>Loading experiment...</div>;
    }

    return <Component {...props} abTest={abTest} />;
  };
}
