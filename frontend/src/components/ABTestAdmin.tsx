/**
 * A/B Test Admin Interface
 *
 * Admin panel for creating and managing A/B tests
 */

import React, { useState } from 'react';
import {
  FiPlus,
  FiPlay,
  FiPause,
  FiStop,
  FiTrash2,
  FiEdit,
  FiEye,
  FiAward,
} from 'react-icons/fi';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useExperiments, type Experiment } from '@/hooks/useABTest';
import { toast } from '@/utils/toast';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface ABTestAdminProps {
  onViewExperiment?: (experimentId: string) => void;
}

const ABTestAdmin: React.FC<ABTestAdminProps> = ({ onViewExperiment }) => {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [statusFilter, setStatusFilter] = useState<string>('');

  const { data: experiments, isLoading } = useExperiments(statusFilter);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading experiments...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">A/B Test Management</h1>
          <p className="text-gray-600 mt-1">Create and manage model comparison experiments</p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="bg-primary-600 hover:bg-primary-700 text-white px-4 py-2 rounded-lg font-medium flex items-center gap-2 transition-colors"
        >
          <FiPlus className="w-5 h-5" />
          New Experiment
        </button>
      </div>

      {/* Stats Cards */}
      <ExperimentStats experiments={experiments || []} />

      {/* Filter */}
      <div className="bg-white rounded-lg shadow-sm p-4">
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-gray-700">Filter by status:</label>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            <option value="">All</option>
            <option value="draft">Draft</option>
            <option value="running">Running</option>
            <option value="paused">Paused</option>
            <option value="completed">Completed</option>
            <option value="cancelled">Cancelled</option>
          </select>
        </div>
      </div>

      {/* Experiments List */}
      <ExperimentsList
        experiments={experiments || []}
        onViewExperiment={onViewExperiment}
      />

      {/* Create Modal */}
      {showCreateModal && (
        <CreateExperimentModal onClose={() => setShowCreateModal(false)} />
      )}
    </div>
  );
};

// ==================== Sub-Components ====================

const ExperimentStats: React.FC<{ experiments: Experiment[] }> = ({ experiments }) => {
  const stats = {
    total: experiments.length,
    running: experiments.filter((e) => e.status === 'running').length,
    completed: experiments.filter((e) => e.status === 'completed').length,
    draft: experiments.filter((e) => e.status === 'draft').length,
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
      <StatCard label="Total Experiments" value={stats.total} color="bg-blue-500" />
      <StatCard label="Running" value={stats.running} color="bg-green-500" />
      <StatCard label="Completed" value={stats.completed} color="bg-purple-500" />
      <StatCard label="Draft" value={stats.draft} color="bg-gray-500" />
    </div>
  );
};

const StatCard: React.FC<{
  label: string;
  value: number;
  color: string;
}> = ({ label, value, color }) => (
  <div className="bg-white rounded-lg shadow-sm p-6">
    <div className={`w-12 h-12 ${color} rounded-lg flex items-center justify-center mb-3`}>
      <span className="text-2xl font-bold text-white">{value}</span>
    </div>
    <p className="text-sm text-gray-600">{label}</p>
  </div>
);

const ExperimentsList: React.FC<{
  experiments: Experiment[];
  onViewExperiment?: (experimentId: string) => void;
}> = ({ experiments, onViewExperiment }) => {
  const queryClient = useQueryClient();

  const startMutation = useMutation({
    mutationFn: async (experimentId: string) => {
      const response = await fetch(
        `${API_BASE_URL}/api/ab-testing/experiments/${experimentId}/start`,
        { method: 'POST' }
      );
      if (!response.ok) throw new Error('Failed to start experiment');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ab-test'] });
      toast.success('Experiment started successfully');
    },
    onError: () => {
      toast.error('Failed to start experiment');
    },
  });

  const pauseMutation = useMutation({
    mutationFn: async (experimentId: string) => {
      const response = await fetch(
        `${API_BASE_URL}/api/ab-testing/experiments/${experimentId}/pause`,
        { method: 'POST' }
      );
      if (!response.ok) throw new Error('Failed to pause experiment');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ab-test'] });
      toast.success('Experiment paused');
    },
    onError: () => {
      toast.error('Failed to pause experiment');
    },
  });

  const stopMutation = useMutation({
    mutationFn: async (experimentId: string) => {
      const response = await fetch(
        `${API_BASE_URL}/api/ab-testing/experiments/${experimentId}/stop`,
        { method: 'POST' }
      );
      if (!response.ok) throw new Error('Failed to stop experiment');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ab-test'] });
      toast.success('Experiment stopped');
    },
    onError: () => {
      toast.error('Failed to stop experiment');
    },
  });

  const deleteMutation = useMutation({
    mutationFn: async (experimentId: string) => {
      const response = await fetch(
        `${API_BASE_URL}/api/ab-testing/experiments/${experimentId}`,
        { method: 'DELETE' }
      );
      if (!response.ok) throw new Error('Failed to delete experiment');
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ab-test'] });
      toast.success('Experiment deleted');
    },
    onError: () => {
      toast.error('Failed to delete experiment');
    },
  });

  if (experiments.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-sm p-12 text-center">
        <p className="text-gray-600 mb-4">No experiments found</p>
        <p className="text-sm text-gray-500">Create your first A/B test to get started</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {experiments.map((experiment) => (
        <div
          key={experiment.id}
          className="bg-white rounded-lg shadow-sm p-6 hover:shadow-md transition-shadow"
        >
          <div className="flex items-start justify-between mb-4">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <h3 className="text-lg font-semibold text-gray-900">{experiment.name}</h3>
                <StatusBadge status={experiment.status} />
              </div>
              <p className="text-gray-600 text-sm mb-3">{experiment.description}</p>
              <div className="flex items-center gap-6 text-sm text-gray-500">
                <span>
                  Created: {new Date(experiment.created_at).toLocaleDateString()}
                </span>
                <span>{experiment.variants.length} variants</span>
                <span>
                  Target: {experiment.target_metric.replace('_', ' ')}
                </span>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {experiment.status === 'draft' && (
                <button
                  onClick={() => startMutation.mutate(experiment.id)}
                  disabled={startMutation.isPending}
                  className="p-2 text-green-600 hover:bg-green-50 rounded-lg transition-colors"
                  title="Start experiment"
                >
                  <FiPlay className="w-5 h-5" />
                </button>
              )}

              {experiment.status === 'running' && (
                <>
                  <button
                    onClick={() => pauseMutation.mutate(experiment.id)}
                    disabled={pauseMutation.isPending}
                    className="p-2 text-yellow-600 hover:bg-yellow-50 rounded-lg transition-colors"
                    title="Pause experiment"
                  >
                    <FiPause className="w-5 h-5" />
                  </button>
                  <button
                    onClick={() => stopMutation.mutate(experiment.id)}
                    disabled={stopMutation.isPending}
                    className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                    title="Stop experiment"
                  >
                    <FiStop className="w-5 h-5" />
                  </button>
                </>
              )}

              {experiment.status === 'paused' && (
                <button
                  onClick={() => startMutation.mutate(experiment.id)}
                  disabled={startMutation.isPending}
                  className="p-2 text-green-600 hover:bg-green-50 rounded-lg transition-colors"
                  title="Resume experiment"
                >
                  <FiPlay className="w-5 h-5" />
                </button>
              )}

              {onViewExperiment && (
                <button
                  onClick={() => onViewExperiment(experiment.id)}
                  className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                  title="View results"
                >
                  <FiEye className="w-5 h-5" />
                </button>
              )}

              {experiment.status !== 'running' && (
                <button
                  onClick={() => {
                    if (confirm('Are you sure you want to delete this experiment?')) {
                      deleteMutation.mutate(experiment.id);
                    }
                  }}
                  disabled={deleteMutation.isPending}
                  className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                  title="Delete experiment"
                >
                  <FiTrash2 className="w-5 h-5" />
                </button>
              )}
            </div>
          </div>

          {/* Variants */}
          <div className="flex gap-2">
            {experiment.variants.map((variant) => (
              <div
                key={variant.id}
                className="flex-1 bg-gray-50 rounded-lg p-3 text-sm"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium text-gray-900">{variant.name}</span>
                  <span className="text-xs text-gray-500">
                    {variant.traffic_percentage}%
                  </span>
                </div>
                <p className="text-xs text-gray-600">{variant.model_name}</p>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

const StatusBadge: React.FC<{ status: string }> = ({ status }) => {
  const colors = {
    draft: 'bg-gray-100 text-gray-800',
    running: 'bg-green-100 text-green-800',
    paused: 'bg-yellow-100 text-yellow-800',
    completed: 'bg-blue-100 text-blue-800',
    cancelled: 'bg-red-100 text-red-800',
  };

  return (
    <span
      className={`px-2 py-1 rounded-full text-xs font-medium ${
        colors[status as keyof typeof colors] || colors.draft
      }`}
    >
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
};

const CreateExperimentModal: React.FC<{ onClose: () => void }> = ({ onClose }) => {
  const queryClient = useQueryClient();

  const [formData, setFormData] = useState({
    name: '',
    description: '',
    target_metric: 'conversion_rate',
    min_sample_size: 100,
    confidence_level: 0.95,
    variants: [
      {
        name: 'Control',
        model_name: 'random_forest',
        traffic_percentage: 50,
        variant_type: 'control',
        description: '',
      },
      {
        name: 'Treatment',
        model_name: 'xgboost',
        traffic_percentage: 50,
        variant_type: 'treatment',
        description: '',
      },
    ],
  });

  const createMutation = useMutation({
    mutationFn: async (data: typeof formData) => {
      const response = await fetch(`${API_BASE_URL}/api/ab-testing/experiments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      if (!response.ok) throw new Error('Failed to create experiment');
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ab-test'] });
      toast.success('Experiment created successfully');
      onClose();
    },
    onError: () => {
      toast.error('Failed to create experiment');
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Validate traffic percentages
    const totalTraffic = formData.variants.reduce(
      (sum, v) => sum + v.traffic_percentage,
      0
    );
    if (Math.abs(totalTraffic - 100) > 0.1) {
      toast.error('Traffic percentages must sum to 100%');
      return;
    }

    createMutation.mutate(formData);
  };

  const addVariant = () => {
    setFormData({
      ...formData,
      variants: [
        ...formData.variants,
        {
          name: `Variant ${formData.variants.length + 1}`,
          model_name: 'logistic_regression',
          traffic_percentage: 0,
          variant_type: 'treatment',
          description: '',
        },
      ],
    });
  };

  const removeVariant = (index: number) => {
    if (formData.variants.length <= 2) {
      toast.warning('Experiment must have at least 2 variants');
      return;
    }
    setFormData({
      ...formData,
      variants: formData.variants.filter((_, i) => i !== index),
    });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-2xl font-bold text-gray-900">Create New Experiment</h2>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {/* Basic Info */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Experiment Name *
            </label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Description *
            </label>
            <textarea
              value={formData.description}
              onChange={(e) =>
                setFormData({ ...formData, description: e.target.value })
              }
              rows={3}
              className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500"
              required
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Target Metric *
              </label>
              <select
                value={formData.target_metric}
                onChange={(e) =>
                  setFormData({ ...formData, target_metric: e.target.value })
                }
                className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="conversion_rate">Conversion Rate</option>
                <option value="confidence">Confidence Score</option>
                <option value="prediction_time">Prediction Time</option>
                <option value="rating">User Rating</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Min Sample Size *
              </label>
              <input
                type="number"
                value={formData.min_sample_size}
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    min_sample_size: parseInt(e.target.value),
                  })
                }
                min={10}
                className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary-500"
                required
              />
            </div>
          </div>

          {/* Variants */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <label className="text-sm font-medium text-gray-700">Variants *</label>
              <button
                type="button"
                onClick={addVariant}
                className="text-primary-600 hover:text-primary-700 text-sm font-medium flex items-center gap-1"
              >
                <FiPlus className="w-4 h-4" />
                Add Variant
              </button>
            </div>

            <div className="space-y-4">
              {formData.variants.map((variant, index) => (
                <div
                  key={index}
                  className="border border-gray-200 rounded-lg p-4 relative"
                >
                  {formData.variants.length > 2 && (
                    <button
                      type="button"
                      onClick={() => removeVariant(index)}
                      className="absolute top-2 right-2 text-red-600 hover:bg-red-50 p-1 rounded"
                    >
                      <FiTrash2 className="w-4 h-4" />
                    </button>
                  )}

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-1">
                        Name
                      </label>
                      <input
                        type="text"
                        value={variant.name}
                        onChange={(e) => {
                          const newVariants = [...formData.variants];
                          newVariants[index].name = e.target.value;
                          setFormData({ ...formData, variants: newVariants });
                        }}
                        className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
                        required
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-1">
                        Model
                      </label>
                      <select
                        value={variant.model_name}
                        onChange={(e) => {
                          const newVariants = [...formData.variants];
                          newVariants[index].model_name = e.target.value;
                          setFormData({ ...formData, variants: newVariants });
                        }}
                        className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
                        required
                      >
                        <option value="random_forest">Random Forest</option>
                        <option value="xgboost">XGBoost</option>
                        <option value="logistic_regression">Logistic Regression</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-1">
                        Traffic %
                      </label>
                      <input
                        type="number"
                        value={variant.traffic_percentage}
                        onChange={(e) => {
                          const newVariants = [...formData.variants];
                          newVariants[index].traffic_percentage = parseFloat(
                            e.target.value
                          );
                          setFormData({ ...formData, variants: newVariants });
                        }}
                        min={0}
                        max={100}
                        step={0.1}
                        className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
                        required
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-gray-600 mb-1">
                        Type
                      </label>
                      <select
                        value={variant.variant_type}
                        onChange={(e) => {
                          const newVariants = [...formData.variants];
                          newVariants[index].variant_type = e.target.value as
                            | 'control'
                            | 'treatment';
                          setFormData({ ...formData, variants: newVariants });
                        }}
                        className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
                      >
                        <option value="control">Control</option>
                        <option value="treatment">Treatment</option>
                      </select>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center justify-end gap-3 pt-4 border-t border-gray-200">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={createMutation.isPending}
              className="bg-primary-600 hover:bg-primary-700 text-white px-6 py-2 rounded-lg font-medium transition-colors disabled:opacity-50"
            >
              {createMutation.isPending ? 'Creating...' : 'Create Experiment'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ABTestAdmin;
