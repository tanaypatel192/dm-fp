import React, { useState, useCallback } from 'react';
import {
  FiInfo,
  FiRefreshCw,
  FiShuffle,
  FiTrash2,
  FiActivity,
} from 'react-icons/fi';
import { Button, LoadingSpinner, ErrorMessage } from '@/components/common';
import type { PatientInput, ComprehensivePredictionOutput } from '@/types/api';
import { predictionApi, handleApiError } from '@/services/api';

// Feature information and normal ranges
const FEATURE_INFO = {
  Pregnancies: {
    label: 'Number of Pregnancies',
    unit: 'count',
    min: 0,
    max: 20,
    step: 1,
    normal: { min: 0, max: 5 },
    description: 'Number of times pregnant. Higher values may increase risk.',
    color: (val: number) =>
      val <= 5 ? 'text-success-600' : val <= 10 ? 'text-warning-600' : 'text-danger-600',
  },
  Glucose: {
    label: 'Plasma Glucose Concentration',
    unit: 'mg/dL',
    min: 0,
    max: 300,
    step: 1,
    normal: { min: 70, max: 100 },
    description: 'Blood glucose level after 2-hour oral glucose tolerance test. Normal fasting: 70-100 mg/dL.',
    color: (val: number) =>
      val < 100 ? 'text-success-600' : val < 126 ? 'text-warning-600' : 'text-danger-600',
  },
  BloodPressure: {
    label: 'Diastolic Blood Pressure',
    unit: 'mm Hg',
    min: 0,
    max: 200,
    step: 1,
    normal: { min: 60, max: 80 },
    description: 'Diastolic (lower) blood pressure. Normal range: 60-80 mm Hg.',
    color: (val: number) =>
      val < 80 ? 'text-success-600' : val < 90 ? 'text-warning-600' : 'text-danger-600',
  },
  SkinThickness: {
    label: 'Triceps Skin Fold Thickness',
    unit: 'mm',
    min: 0,
    max: 100,
    step: 1,
    normal: { min: 10, max: 30 },
    description: 'Thickness of skin fold at triceps. Indicates body fat percentage.',
    color: (val: number) =>
      val <= 30 ? 'text-success-600' : val <= 50 ? 'text-warning-600' : 'text-danger-600',
  },
  Insulin: {
    label: '2-Hour Serum Insulin',
    unit: 'μU/mL',
    min: 0,
    max: 900,
    step: 1,
    normal: { min: 16, max: 166 },
    description: 'Insulin level measured 2 hours after glucose load. Normal: 16-166 μU/mL.',
    color: (val: number) =>
      val <= 166 ? 'text-success-600' : val <= 300 ? 'text-warning-600' : 'text-danger-600',
  },
  BMI: {
    label: 'Body Mass Index',
    unit: 'kg/m²',
    min: 0,
    max: 70,
    step: 0.1,
    normal: { min: 18.5, max: 24.9 },
    description: 'Weight (kg) / Height (m)². Normal: 18.5-24.9. Overweight: 25-29.9. Obese: ≥30.',
    color: (val: number) =>
      val < 25 ? 'text-success-600' : val < 30 ? 'text-warning-600' : 'text-danger-600',
  },
  DiabetesPedigreeFunction: {
    label: 'Diabetes Pedigree Function',
    unit: 'score',
    min: 0,
    max: 3,
    step: 0.001,
    normal: { min: 0, max: 0.5 },
    description: 'Genetic likelihood of diabetes based on family history. Higher = stronger family history.',
    color: (val: number) =>
      val < 0.5 ? 'text-success-600' : val < 1.0 ? 'text-warning-600' : 'text-danger-600',
  },
  Age: {
    label: 'Age',
    unit: 'years',
    min: 1,
    max: 120,
    step: 1,
    normal: { min: 21, max: 45 },
    description: 'Age in years. Risk increases with age, especially after 45.',
    color: (val: number) =>
      val < 45 ? 'text-success-600' : val < 65 ? 'text-warning-600' : 'text-danger-600',
  },
};

// Example patient data
const EXAMPLE_PATIENTS = {
  highRisk: {
    Pregnancies: 6,
    Glucose: 148,
    BloodPressure: 72,
    SkinThickness: 35,
    Insulin: 0,
    BMI: 33.6,
    DiabetesPedigreeFunction: 0.627,
    Age: 50,
  },
  lowRisk: {
    Pregnancies: 1,
    Glucose: 85,
    BloodPressure: 66,
    SkinThickness: 29,
    Insulin: 0,
    BMI: 26.6,
    DiabetesPedigreeFunction: 0.351,
    Age: 31,
  },
  moderate: {
    Pregnancies: 3,
    Glucose: 110,
    BloodPressure: 75,
    SkinThickness: 25,
    Insulin: 120,
    BMI: 28.5,
    DiabetesPedigreeFunction: 0.45,
    Age: 42,
  },
};

interface PredictionFormProps {
  onPredictionComplete?: (result: ComprehensivePredictionOutput) => void;
  onPredictionStart?: () => void;
}

interface FormErrors {
  [key: string]: string;
}

const PredictionForm: React.FC<PredictionFormProps> = ({
  onPredictionComplete,
  onPredictionStart,
}) => {
  const [formData, setFormData] = useState<PatientInput>({
    Pregnancies: 1,
    Glucose: 120,
    BloodPressure: 70,
    SkinThickness: 20,
    Insulin: 79,
    BMI: 32,
    DiabetesPedigreeFunction: 0.472,
    Age: 33,
  });

  const [errors, setErrors] = useState<FormErrors>({});
  const [touched, setTouched] = useState<{ [key: string]: boolean }>({});
  const [loading, setLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [showTooltip, setShowTooltip] = useState<string | null>(null);

  // Validation
  const validateField = useCallback((name: keyof PatientInput, value: number): string => {
    const info = FEATURE_INFO[name];
    if (value < info.min || value > info.max) {
      return `Value must be between ${info.min} and ${info.max}`;
    }
    return '';
  }, []);

  const validateForm = useCallback((): boolean => {
    const newErrors: FormErrors = {};
    Object.keys(formData).forEach((key) => {
      const error = validateField(key as keyof PatientInput, formData[key as keyof PatientInput]);
      if (error) {
        newErrors[key] = error;
      }
    });
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [formData, validateField]);

  // Handle input change
  const handleInputChange = (name: keyof PatientInput, value: number) => {
    setFormData((prev) => ({ ...prev, [name]: value }));
    setTouched((prev) => ({ ...prev, [name]: true }));

    // Validate on change
    const error = validateField(name, value);
    setErrors((prev) => {
      const newErrors = { ...prev };
      if (error) {
        newErrors[name] = error;
      } else {
        // Remove the error if validation passes
        delete newErrors[name];
      }
      return newErrors;
    });
  };

  // Load example patient
  const loadExample = (type: keyof typeof EXAMPLE_PATIENTS) => {
    setFormData(EXAMPLE_PATIENTS[type]);
    setErrors({});
    setTouched({});
    setApiError(null);
  };

  // Generate random patient
  const generateRandom = () => {
    const random: PatientInput = {
      Pregnancies: Math.floor(Math.random() * 15),
      Glucose: Math.floor(Math.random() * 150) + 50,
      BloodPressure: Math.floor(Math.random() * 60) + 50,
      SkinThickness: Math.floor(Math.random() * 50) + 10,
      Insulin: Math.floor(Math.random() * 400),
      BMI: Math.floor(Math.random() * 30) + 18,
      DiabetesPedigreeFunction: Math.random() * 2,
      Age: Math.floor(Math.random() * 70) + 20,
    };
    setFormData(random);
    setErrors({});
    setTouched({});
    setApiError(null);
  };

  // Clear form
  const clearForm = () => {
    setFormData({
      Pregnancies: 0,
      Glucose: 100,
      BloodPressure: 70,
      SkinThickness: 20,
      Insulin: 80,
      BMI: 25,
      DiabetesPedigreeFunction: 0.5,
      Age: 30,
    });
    setErrors({});
    setTouched({});
    setApiError(null);
  };

  // Submit form
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setLoading(true);
    setApiError(null);

    if (onPredictionStart) {
      onPredictionStart();
    }

    try {
      const result = await predictionApi.predictExplain(formData);

      if (onPredictionComplete) {
        onPredictionComplete(result);
      }
    } catch (error) {
      const errorMessage = handleApiError(error);
      setApiError(errorMessage);
      console.error('Prediction error:', error);
    } finally {
      setLoading(false);
    }
  };

  // Get progress percentage for visual indicator
  const getProgressPercentage = (name: keyof PatientInput, value: number): number => {
    const info = FEATURE_INFO[name];
    return ((value - info.min) / (info.max - info.min)) * 100;
  };

  // Get normal range percentage for visual indicator
  const getNormalRangePosition = (name: keyof PatientInput) => {
    const info = FEATURE_INFO[name];
    const start = ((info.normal.min - info.min) / (info.max - info.min)) * 100;
    const end = ((info.normal.max - info.min) / (info.max - info.min)) * 100;
    return { start, width: end - start };
  };

  // Render input field
  const renderField = (name: keyof PatientInput) => {
    const info = FEATURE_INFO[name];
    const value = formData[name];
    const error = touched[name] ? errors[name] : '';
    const progress = getProgressPercentage(name, value);
    const normalRange = getNormalRangePosition(name);
    const colorClass = info.color(value);

    return (
      <div key={name} className="space-y-2">
        {/* Label with tooltip */}
        <div className="flex items-center justify-between">
          <label className="label flex items-center gap-2">
            <span>{info.label}</span>
            <button
              type="button"
              onMouseEnter={() => setShowTooltip(name)}
              onMouseLeave={() => setShowTooltip(null)}
              className="relative"
            >
              <FiInfo className="w-4 h-4 text-gray-400 hover:text-primary-600 transition-colors" />
              {showTooltip === name && (
                <div className="absolute left-0 top-6 z-10 w-64 p-3 bg-gray-900 dark:bg-gray-700 text-white text-xs rounded-lg shadow-lg">
                  {info.description}
                  <div className="absolute -top-1 left-4 w-2 h-2 bg-gray-900 dark:bg-gray-700 transform rotate-45" />
                </div>
              )}
            </button>
          </label>
          <span className={`text-sm font-semibold ${colorClass}`}>
            {value} {info.unit}
          </span>
        </div>

        {/* Dual input: Slider + Number */}
        <div className="grid grid-cols-[1fr_auto] gap-3 items-center">
          {/* Slider with visual indicators */}
          <div className="relative">
            {/* Normal range indicator */}
            <div
              className="absolute top-2 h-2 bg-success-200 dark:bg-success-900/30 rounded-full pointer-events-none"
              style={{
                left: `${normalRange.start}%`,
                width: `${normalRange.width}%`,
              }}
            />

            {/* Slider */}
            <input
              type="range"
              min={info.min}
              max={info.max}
              step={info.step}
              value={value}
              onChange={(e) => handleInputChange(name, parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-primary-600 relative z-10"
            />

            {/* Progress indicator */}
            <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
              <span>{info.min}</span>
              <span className="text-xs text-gray-600 dark:text-gray-400">
                Normal: {info.normal.min}-{info.normal.max}
              </span>
              <span>{info.max}</span>
            </div>
          </div>

          {/* Number input */}
          <input
            type="number"
            min={info.min}
            max={info.max}
            step={info.step}
            value={value}
            onChange={(e) => {
              const val = e.target.value === '' ? info.min : parseFloat(e.target.value);
              if (!isNaN(val)) {
                handleInputChange(name, val);
              }
            }}
            className={`input w-24 text-center ${error ? 'border-danger-500 focus:ring-danger-500' : ''
              }`}
          />
        </div>

        {/* Mini visualization bar */}
        <div className="h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-300 ${value < info.normal.max
              ? 'bg-success-500'
              : value < info.max * 0.7
                ? 'bg-warning-500'
                : 'bg-danger-500'
              }`}
            style={{ width: `${Math.min(progress, 100)}%` }}
          />
        </div>

        {/* Error message */}
        {error && <p className="text-xs text-danger-600 dark:text-danger-400">{error}</p>}
      </div>
    );
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Action buttons */}
      <div className="flex flex-wrap gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => loadExample('highRisk')}
          icon={<FiActivity />}
        >
          High Risk Example
        </Button>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => loadExample('lowRisk')}
          icon={<FiActivity />}
        >
          Low Risk Example
        </Button>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => loadExample('moderate')}
          icon={<FiActivity />}
        >
          Moderate Example
        </Button>
        <Button
          type="button"
          variant="secondary"
          size="sm"
          onClick={generateRandom}
          icon={<FiShuffle />}
        >
          Random
        </Button>
        <Button
          type="button"
          variant="secondary"
          size="sm"
          onClick={clearForm}
          icon={<FiTrash2 />}
        >
          Clear
        </Button>
      </div>

      {/* Error message */}
      {apiError && (
        <ErrorMessage
          title="Prediction Failed"
          message={apiError}
          variant="error"
          onDismiss={() => setApiError(null)}
        />
      )}

      {/* Form fields in a grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {(Object.keys(FEATURE_INFO) as Array<keyof PatientInput>).map((name) =>
          renderField(name)
        )}
      </div>

      {/* Submit button */}
      <div className="flex justify-end pt-4 border-t border-gray-200 dark:border-gray-700">
        <Button
          type="submit"
          variant="primary"
          size="lg"
          loading={loading}
          disabled={loading || Object.values(errors).some(err => err !== '')}
          icon={<FiRefreshCw />}
        >
          {loading ? 'Analyzing...' : 'Predict Diabetes Risk'}
        </Button>
      </div>
    </form>
  );
};

export default PredictionForm;
