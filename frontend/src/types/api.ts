/**
 * Type definitions for API requests and responses
 */

// Patient Input
export interface PatientInput {
  Pregnancies: number;
  Glucose: number;
  BloodPressure: number;
  SkinThickness: number;
  Insulin: number;
  BMI: number;
  DiabetesPedigreeFunction: number;
  Age: number;
}

// Prediction Output
export interface PredictionOutput {
  prediction: number;
  prediction_label: string;
  probability: number;
  risk_level: string;
  confidence: number;
  model_used: string;
}

// Batch Prediction
export interface BatchPatientInput {
  patients: PatientInput[];
}

export interface BatchPredictionOutput {
  predictions: PredictionOutput[];
  total_processed: number;
  processing_time_ms: number;
}

// Model Information
export interface ModelMetrics {
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
  is_loaded: boolean;
  last_updated: string | null;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  rank: number;
}

// Model Comparison
export interface ModelComparisonOutput {
  input_data: Record<string, any>;
  predictions: Record<string, PredictionOutput>;
  consensus_prediction: number;
  consensus_label: string;
  agreement_percentage: number;
}

// Data Statistics
export interface DataStats {
  total_samples: number;
  features_count: number;
  class_distribution: Record<string, number>;
  feature_statistics: Record<string, FeatureStats>;
}

export interface FeatureStats {
  mean: number;
  std: number;
  min: number;
  max: number;
  median: number;
}

// Health Check
export interface HealthResponse {
  status: string;
  timestamp: string;
  models_loaded: number;
  available_models: string[];
}

// Comprehensive Prediction (with explanations)
export interface FeatureContribution {
  feature: string;
  value: number;
  contribution: number;
  impact: string; // "increases" or "decreases"
}

export interface SHAPExplanation {
  base_value: number;
  prediction_value: number;
  feature_contributions: FeatureContribution[];
  top_features: string[];
}

export interface ModelPredictionDetail {
  model_name: string;
  prediction: number;
  prediction_label: string;
  probability: number;
  confidence: number;
}

export interface SimilarPatient {
  similarity_score: number;
  outcome: number;
  outcome_label: string;
  key_similarities: string[];
}

export interface RiskFactor {
  factor: string;
  current_value: number;
  risk_level: string;
  is_modifiable: boolean;
}

export interface Recommendation {
  category: string;
  priority: string; // "High", "Medium", "Low"
  recommendation: string;
  rationale: string;
}

export interface ComprehensivePredictionOutput {
  // Patient data
  input_data: Record<string, any>;

  // Predictions from all models
  model_predictions: ModelPredictionDetail[];

  // Ensemble prediction
  ensemble_prediction: number;
  ensemble_label: string;
  ensemble_probability: number;
  ensemble_confidence: number;

  // Risk assessment
  risk_level: string;
  risk_score: number;

  // SHAP explanations
  shap_available: boolean;
  shap_explanation: SHAPExplanation | null;

  // Risk factors
  risk_factors: RiskFactor[];

  // Similar patients
  similar_patients: SimilarPatient[];

  // Personalized recommendations
  recommendations: Recommendation[];

  // Metadata
  processing_time_ms: number;
  timestamp: string;
}

// Error Response
export interface ErrorResponse {
  error: string;
  status_code: number;
  timestamp: string;
}
