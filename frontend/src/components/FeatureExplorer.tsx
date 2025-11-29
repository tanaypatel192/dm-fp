import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  FiActivity,
  FiTrendingUp,
  FiSave,
  FiRefreshCw,
  FiTrash2,
  FiZap,
  FiAlertCircle,
} from 'react-icons/fi';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler } from 'chart.js';
import { Card, Button, LoadingSpinner } from '@/components/common';
import { predictionApi, handleApiError } from '@/services/api';
import type { PatientInput, ComprehensivePredictionOutput } from '@/types/api';

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

interface FeatureRange {
  min: number;
  max: number;
  step: number;
  unit: string;
  normal: { min: number; max: number };
}

const FEATURE_RANGES: Record<keyof PatientInput, FeatureRange> = {
  Pregnancies: { min: 0, max: 20, step: 1, unit: 'count', normal: { min: 0, max: 5 } },
  Glucose: { min: 0, max: 300, step: 1, unit: 'mg/dL', normal: { min: 70, max: 100 } },
  BloodPressure: { min: 0, max: 200, step: 1, unit: 'mm Hg', normal: { min: 60, max: 80 } },
  SkinThickness: { min: 0, max: 100, step: 1, unit: 'mm', normal: { min: 10, max: 30 } },
  Insulin: { min: 0, max: 900, step: 1, unit: 'μU/mL', normal: { min: 16, max: 166 } },
  BMI: { min: 0, max: 70, step: 0.1, unit: 'kg/m²', normal: { min: 18.5, max: 24.9 } },
  DiabetesPedigreeFunction: { min: 0, max: 3, step: 0.01, unit: 'score', normal: { min: 0, max: 0.5 } },
  Age: { min: 1, max: 120, step: 1, unit: 'years', normal: { min: 21, max: 45 } },
};

interface Scenario {
  id: string;
  name: string;
  values: PatientInput;
  prediction: number;
  riskLevel: string;
}

const FeatureExplorer: React.FC = () => {
  // Core state
  const [features, setFeatures] = useState<PatientInput>({
    Pregnancies: 3,
    Glucose: 120,
    BloodPressure: 70,
    SkinThickness: 20,
    Insulin: 80,
    BMI: 28,
    DiabetesPedigreeFunction: 0.5,
    Age: 35,
  });

  const [prediction, setPrediction] = useState<ComprehensivePredictionOutput | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Feature analysis state
  const [analyzingFeature, setAnalyzingFeature] = useState<keyof PatientInput | null>(null);
  const [featureAnalysisData, setFeatureAnalysisData] = useState<{ x: number[]; y: number[] } | null>(null);
  const [analyzingLoading, setAnalyzingLoading] = useState(false);

  // Scenario comparison state
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [nextScenarioId, setNextScenarioId] = useState(1);

  // Debounce timer ref
  const debounceTimer = useRef<NodeJS.Timeout | null>(null);

  // Debounced prediction update
  const updatePrediction = useCallback(async (values: PatientInput) => {
    try {
      setLoading(true);
      setError(null);
      const result = await predictionApi.predictExplain(values);
      setPrediction(result);
    } catch (err) {
      setError(handleApiError(err));
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Debounced update with 500ms delay
  useEffect(() => {
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }

    debounceTimer.current = setTimeout(() => {
      updatePrediction(features);
    }, 500);

    return () => {
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current);
      }
    };
  }, [features, updatePrediction]);

  // Handle feature change
  const handleFeatureChange = (feature: keyof PatientInput, value: number) => {
    setFeatures((prev) => ({ ...prev, [feature]: value }));
  };

  // What-if scenarios
  const applyWhatIf = (scenario: 'loseWeight' | 'lowerGlucose' | 'younger') => {
    setFeatures((prev) => {
      const updated = { ...prev };
      switch (scenario) {
        case 'loseWeight':
          // Lose 10 lbs ≈ reduce BMI by ~1.5
          updated.BMI = Math.max(15, prev.BMI - 1.5);
          break;
        case 'lowerGlucose':
          // Lower glucose by 20 mg/dL
          updated.Glucose = Math.max(70, prev.Glucose - 20);
          break;
        case 'younger':
          // 5 years younger
          updated.Age = Math.max(18, prev.Age - 5);
          break;
      }
      return updated;
    });
  };

  // Save current scenario
  const saveScenario = () => {
    if (scenarios.length >= 3) {
      alert('Maximum 3 scenarios allowed. Delete one to add another.');
      return;
    }

    if (!prediction) return;

    const newScenario: Scenario = {
      id: `scenario-${nextScenarioId}`,
      name: `Scenario ${nextScenarioId}`,
      values: { ...features },
      prediction: prediction.ensemble_probability,
      riskLevel: prediction.risk_level,
    };

    setScenarios((prev) => [...prev, newScenario]);
    setNextScenarioId((prev) => prev + 1);
  };

  // Delete scenario
  const deleteScenario = (id: string) => {
    setScenarios((prev) => prev.filter((s) => s.id !== id));
  };

  // Load scenario
  const loadScenario = (scenario: Scenario) => {
    setFeatures(scenario.values);
  };

  // Analyze feature
  const analyzeFeature = async (feature: keyof PatientInput) => {
    setAnalyzingFeature(feature);
    setAnalyzingLoading(true);

    try {
      const range = FEATURE_RANGES[feature];
      const steps = 20;
      const stepSize = (range.max - range.min) / steps;

      const xValues: number[] = [];
      const yValues: number[] = [];

      // Keep other features constant
      const baseFeatures = { ...features };

      for (let i = 0; i <= steps; i++) {
        const value = range.min + i * stepSize;
        xValues.push(value);

        const testFeatures = { ...baseFeatures, [feature]: value };
        const result = await predictionApi.predictExplain(testFeatures);
        yValues.push(result.ensemble_probability * 100);
      }

      setFeatureAnalysisData({ x: xValues, y: yValues });
    } catch (err) {
      console.error('Feature analysis error:', err);
    } finally {
      setAnalyzingLoading(false);
    }
  };

  // Get risk color
  const getRiskColor = (probability: number) => {
    if (probability < 0.3) return { bg: 'bg-success-500', text: 'text-success-600', glow: 'shadow-success-500/50' };
    if (probability < 0.7) return { bg: 'bg-warning-500', text: 'text-warning-600', glow: 'shadow-warning-500/50' };
    return { bg: 'bg-danger-500', text: 'text-danger-600', glow: 'shadow-danger-500/50' };
  };

  // Risk meter SVG gauge
  const RiskMeter = ({ probability }: { probability: number }) => {
    const angle = -90 + probability * 180; // -90 to 90 degrees
    const riskColor = getRiskColor(probability);

    return (
      <div className="relative w-64 h-32 mx-auto">
        <svg viewBox="0 0 200 100" className="w-full h-full">
          {/* Background arc */}
          <path
            d="M 20 80 A 80 80 0 0 1 180 80"
            fill="none"
            stroke="#e5e7eb"
            strokeWidth="15"
            strokeLinecap="round"
          />
          {/* Colored arc */}
          <path
            d="M 20 80 A 80 80 0 0 1 180 80"
            fill="none"
            stroke={probability < 0.3 ? '#22c55e' : probability < 0.7 ? '#f59e0b' : '#ef4444'}
            strokeWidth="15"
            strokeLinecap="round"
            strokeDasharray={`${probability * 251} 251`}
            className="transition-all duration-700 ease-in-out"
          />
          {/* Needle */}
          <g transform={`rotate(${angle} 100 80)`}>
            <line
              x1="100"
              y1="80"
              x2="100"
              y2="25"
              stroke="#1f2937"
              strokeWidth="3"
              strokeLinecap="round"
            />
            <circle cx="100" cy="80" r="6" fill="#1f2937" />
          </g>
          {/* Labels */}
          <text x="25" y="95" className="text-xs fill-success-600" fontSize="10">Low</text>
          <text x="90" y="20" className="text-xs fill-warning-600" fontSize="10">Med</text>
          <text x="165" y="95" className="text-xs fill-danger-600" fontSize="10">High</text>
        </svg>
      </div>
    );
  };

  // Partial dependence chart
  const featureAnalysisChart = useMemo(() => {
    if (!featureAnalysisData || !analyzingFeature) return null;

    const currentValue = features[analyzingFeature];
    const currentIndex = featureAnalysisData.x.findIndex((x) => x >= currentValue);

    return {
      labels: featureAnalysisData.x.map((x) => x.toFixed(1)),
      datasets: [
        {
          label: 'Diabetes Risk (%)',
          data: featureAnalysisData.y,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderWidth: 3,
          fill: true,
          tension: 0.4,
          pointRadius: featureAnalysisData.x.map((_, i) => (i === currentIndex ? 8 : 3)),
          pointBackgroundColor: featureAnalysisData.x.map((_, i) =>
            i === currentIndex ? '#ef4444' : '#3b82f6'
          ),
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
        },
      ],
    };
  }, [featureAnalysisData, analyzingFeature, features]);

  return (
    <div className="space-y-6">
      {/* Current Prediction Display */}
      <Card className={`relative overflow-hidden transition-all duration-500 ${
        prediction ? getRiskColor(prediction.ensemble_probability).glow : ''
      }`}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Left: Probability */}
          <div className="text-center">
            <div className="text-sm text-gray-600 dark:text-gray-400 mb-2">Current Risk</div>
            <div className={`text-7xl font-bold transition-all duration-700 ${
              prediction ? getRiskColor(prediction.ensemble_probability).text : 'text-gray-400'
            }`}>
              {loading ? (
                <LoadingSpinner size="lg" />
              ) : prediction ? (
                `${(prediction.ensemble_probability * 100).toFixed(1)}%`
              ) : (
                '--'
              )}
            </div>
            <div className={`text-xl font-semibold mt-2 transition-all duration-700 ${
              prediction ? getRiskColor(prediction.ensemble_probability).text : 'text-gray-400'
            }`}>
              {prediction ? prediction.risk_level : 'No Data'}
            </div>
          </div>

          {/* Right: Risk Meter */}
          <div className="flex items-center justify-center">
            {prediction && <RiskMeter probability={prediction.ensemble_probability} />}
          </div>
        </div>

        {/* Live update indicator */}
        {loading && (
          <div className="absolute top-2 right-2">
            <div className="flex items-center gap-2 text-sm text-primary-600 dark:text-primary-400">
              <div className="w-2 h-2 bg-primary-600 rounded-full animate-pulse" />
              Updating...
            </div>
          </div>
        )}
      </Card>

      {/* What-If Scenarios */}
      <Card title="Quick What-If Scenarios" subtitle="See how simple changes affect your risk">
        <div className="flex flex-wrap gap-3">
          <Button
            variant="outline"
            size="md"
            onClick={() => applyWhatIf('loseWeight')}
            icon={<FiZap />}
          >
            Lose 10 lbs
          </Button>
          <Button
            variant="outline"
            size="md"
            onClick={() => applyWhatIf('lowerGlucose')}
            icon={<FiZap />}
          >
            Lower Glucose by 20
          </Button>
          <Button
            variant="outline"
            size="md"
            onClick={() => applyWhatIf('younger')}
            icon={<FiZap />}
          >
            5 Years Younger
          </Button>
          <Button
            variant="primary"
            size="md"
            onClick={saveScenario}
            icon={<FiSave />}
            disabled={!prediction || scenarios.length >= 3}
          >
            Save Scenario ({scenarios.length}/3)
          </Button>
        </div>
      </Card>

      {/* Interactive Sliders */}
      <Card title="Adjust Patient Features" subtitle="Move sliders to see real-time predictions">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {(Object.keys(FEATURE_RANGES) as Array<keyof PatientInput>).map((feature) => {
            const range = FEATURE_RANGES[feature];
            const value = features[feature];
            const percentage = ((value - range.min) / (range.max - range.min)) * 100;

            return (
              <div key={feature} className="space-y-2">
                <div className="flex justify-between items-center">
                  <label className="text-sm font-medium">
                    {feature.replace(/([A-Z])/g, ' $1').trim()}
                  </label>
                  <div className="flex items-center gap-2">
                    <span className="text-lg font-bold">
                      {value.toFixed(range.step < 1 ? 2 : 0)}
                    </span>
                    <span className="text-xs text-gray-500">{range.unit}</span>
                  </div>
                </div>

                {/* Slider with background */}
                <div className="relative">
                  <div
                    className="absolute top-2 h-2 bg-success-200 dark:bg-success-900/30 rounded-full"
                    style={{
                      left: `${((range.normal.min - range.min) / (range.max - range.min)) * 100}%`,
                      width: `${((range.normal.max - range.normal.min) / (range.max - range.min)) * 100}%`,
                    }}
                  />
                  <input
                    type="range"
                    min={range.min}
                    max={range.max}
                    step={range.step}
                    value={value}
                    onChange={(e) => handleFeatureChange(feature, parseFloat(e.target.value))}
                    className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-primary-600 relative z-10"
                  />
                </div>

                {/* Progress bar */}
                <div className="h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-300 ${
                      value <= range.normal.max ? 'bg-success-500' :
                      value <= range.max * 0.7 ? 'bg-warning-500' :
                      'bg-danger-500'
                    }`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>

                {/* Analyze button */}
                <button
                  onClick={() => analyzeFeature(feature)}
                  className="text-xs text-primary-600 hover:text-primary-700 dark:text-primary-400 transition-colors"
                >
                  Analyze this feature →
                </button>
              </div>
            );
          })}
        </div>
      </Card>

      {/* Feature Analysis */}
      {analyzingFeature && (
        <Card
          title={`Feature Analysis: ${analyzingFeature}`}
          subtitle="How this feature affects diabetes risk across its full range"
        >
          {analyzingLoading ? (
            <div className="flex items-center justify-center py-12">
              <LoadingSpinner size="lg" text="Analyzing feature..." />
            </div>
          ) : featureAnalysisChart ? (
            <>
              <div className="h-64">
                <Line
                  data={featureAnalysisChart}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        display: false,
                      },
                      tooltip: {
                        callbacks: {
                          label: (context) => `Risk: ${context.parsed.y.toFixed(1)}%`,
                        },
                      },
                    },
                    scales: {
                      x: {
                        title: {
                          display: true,
                          text: `${analyzingFeature} (${FEATURE_RANGES[analyzingFeature].unit})`,
                        },
                      },
                      y: {
                        title: {
                          display: true,
                          text: 'Diabetes Risk (%)',
                        },
                        min: 0,
                        max: 100,
                      },
                    },
                  }}
                />
              </div>
              <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  <strong>Current value</strong> is marked with a red dot. This shows how your risk
                  would change if you modified this feature while keeping others constant.
                </p>
              </div>
            </>
          ) : null}
        </Card>
      )}

      {/* Scenario Comparison */}
      {scenarios.length > 0 && (
        <Card
          title="Saved Scenarios"
          subtitle="Compare different scenarios side-by-side"
        >
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {scenarios.map((scenario) => {
              const riskColor = getRiskColor(scenario.prediction);
              const isLowest = scenario.prediction === Math.min(...scenarios.map((s) => s.prediction));

              return (
                <div
                  key={scenario.id}
                  className={`p-4 border-2 rounded-lg transition-all ${
                    isLowest
                      ? 'border-success-500 bg-success-50 dark:bg-success-900/20'
                      : 'border-gray-200 dark:border-gray-700'
                  }`}
                >
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <h4 className="font-semibold">{scenario.name}</h4>
                      {isLowest && (
                        <span className="text-xs text-success-600 dark:text-success-400">
                          ✓ Lowest Risk
                        </span>
                      )}
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={() => loadScenario(scenario)}
                        className="text-xs text-primary-600 hover:text-primary-700"
                        title="Load scenario"
                      >
                        <FiRefreshCw className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => deleteScenario(scenario.id)}
                        className="text-xs text-danger-600 hover:text-danger-700"
                        title="Delete scenario"
                      >
                        <FiTrash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>

                  <div className={`text-3xl font-bold mb-2 ${riskColor.text}`}>
                    {(scenario.prediction * 100).toFixed(1)}%
                  </div>

                  <div className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                    <div>BMI: {scenario.values.BMI.toFixed(1)}</div>
                    <div>Glucose: {scenario.values.Glucose}</div>
                    <div>Age: {scenario.values.Age}</div>
                  </div>
                </div>
              );
            })}
          </div>

          {scenarios.length >= 2 && (
            <div className="mt-4 p-4 bg-primary-50 dark:bg-primary-900/20 rounded-lg">
              <h5 className="font-semibold mb-2">Comparison Insights</h5>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Risk difference: {((Math.max(...scenarios.map((s) => s.prediction)) -
                  Math.min(...scenarios.map((s) => s.prediction))) * 100).toFixed(1)}%
                between scenarios
              </p>
            </div>
          )}
        </Card>
      )}

      {/* Error Display */}
      {error && (
        <div className="p-4 bg-danger-50 dark:bg-danger-900/20 border border-danger-200 dark:border-danger-800 rounded-lg">
          <div className="flex items-center gap-2 text-danger-600 dark:text-danger-400">
            <FiAlertCircle className="w-5 h-5" />
            <span className="font-semibold">Error:</span>
            <span>{error}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default FeatureExplorer;
