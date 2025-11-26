import React, { useState, useEffect, useMemo } from 'react';
import Plot from 'react-plotly.js';
import {
  FiDownload,
  FiRefreshCw,
  FiBarChart2,
  FiGrid,
  FiLayers,
  FiActivity,
} from 'react-icons/fi';
import { Button, Card, LoadingSpinner, ErrorMessage } from '@/components/common';
import { dataApi, handleApiError } from '@/services/api';
import type { DataStats } from '@/types/api';

// Feature names
const FEATURES = [
  'Pregnancies',
  'Glucose',
  'BloodPressure',
  'SkinThickness',
  'Insulin',
  'BMI',
  'DiabetesPedigreeFunction',
  'Age',
];

// Color schemes for charts
const COLOR_SCHEMES = {
  viridis: { name: 'Viridis', colors: ['#440154', '#31688e', '#35b779', '#fde724'] },
  plasma: { name: 'Plasma', colors: ['#0d0887', '#7e03a8', '#cc4778', '#f89540', '#f0f921'] },
  cool: { name: 'Cool', colors: ['#0000ff', '#00ffff', '#00ff00'] },
  warm: { name: 'Warm', colors: ['#ff0000', '#ff7f00', '#ffff00'] },
  earth: { name: 'Earth', colors: ['#654321', '#8b4513', '#a0522d', '#cd853f', '#deb887'] },
};

// Mock data generator (in real app, this would come from backend)
const generateMockDistribution = (feature: string, outcome: number) => {
  const baseValues: Record<string, { mean: number; std: number }> = {
    Pregnancies: { mean: outcome ? 4.5 : 3.0, std: 3.5 },
    Glucose: { mean: outcome ? 140 : 110, std: 30 },
    BloodPressure: { mean: 70, std: 12 },
    SkinThickness: { mean: 20, std: 15 },
    Insulin: { mean: outcome ? 120 : 80, std: 115 },
    BMI: { mean: outcome ? 33 : 30, std: 7 },
    DiabetesPedigreeFunction: { mean: outcome ? 0.5 : 0.3, std: 0.3 },
    Age: { mean: outcome ? 38 : 30, std: 12 },
  };

  const { mean, std } = baseValues[feature] || { mean: 0, std: 1 };
  const data: number[] = [];

  for (let i = 0; i < 100; i++) {
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    data.push(Math.max(0, mean + z * std));
  }

  return data;
};

// Generate mock correlation matrix
const generateMockCorrelation = () => {
  const matrix: number[][] = [];
  for (let i = 0; i < FEATURES.length; i++) {
    matrix[i] = [];
    for (let j = 0; j < FEATURES.length; j++) {
      if (i === j) {
        matrix[i][j] = 1;
      } else {
        matrix[i][j] = (Math.random() - 0.5) * 2;
      }
    }
  }
  return matrix;
};

const VisualizationDashboard: React.FC = () => {
  // State
  const [dataStats, setDataStats] = useState<DataStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Control state
  const [selectedFeature, setSelectedFeature] = useState<string>('Glucose');
  const [selectedFeatureX, setSelectedFeatureX] = useState<string>('Glucose');
  const [selectedFeatureY, setSelectedFeatureY] = useState<string>('BMI');
  const [selectedFeatureZ, setSelectedFeatureZ] = useState<string>('Age');
  const [selectedPairFeatures, setSelectedPairFeatures] = useState<string[]>([
    'Glucose',
    'BMI',
    'Age',
  ]);
  const [colorScheme, setColorScheme] = useState<keyof typeof COLOR_SCHEMES>('viridis');
  const [outcomeFilter, setOutcomeFilter] = useState<'all' | 'diabetic' | 'non-diabetic'>('all');

  // Fetch data on mount
  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const stats = await dataApi.getStats();
      setDataStats(stats);
    } catch (err) {
      setError(handleApiError(err));
    } finally {
      setLoading(false);
    }
  };

  // Calculate summary statistics
  const summaryStats = useMemo(() => {
    if (!dataStats) return null;

    const totalPatients = dataStats.total_samples;
    const diabeticCount = dataStats.class_distribution['1'] || 0;
    const diabeticPercentage = (diabeticCount / totalPatients) * 100;
    const avgAge = dataStats.feature_statistics['Age']?.mean || 0;
    const avgBMI = dataStats.feature_statistics['BMI']?.mean || 0;

    return {
      totalPatients,
      diabeticCount,
      diabeticPercentage,
      avgAge,
      avgBMI,
    };
  }, [dataStats]);

  // Generate feature distribution data
  const featureDistributionData = useMemo(() => {
    const nonDiabeticData = generateMockDistribution(selectedFeature, 0);
    const diabeticData = generateMockDistribution(selectedFeature, 1);

    return {
      nonDiabetic: nonDiabeticData,
      diabetic: diabeticData,
      combined: [...nonDiabeticData, ...diabeticData],
    };
  }, [selectedFeature, outcomeFilter]);

  // Generate correlation matrix
  const correlationMatrix = useMemo(() => generateMockCorrelation(), []);

  // Generate 3D scatter data
  const scatter3DData = useMemo(() => {
    const diabeticPoints = {
      x: generateMockDistribution(selectedFeatureX, 1),
      y: generateMockDistribution(selectedFeatureY, 1),
      z: generateMockDistribution(selectedFeatureZ, 1),
    };
    const nonDiabeticPoints = {
      x: generateMockDistribution(selectedFeatureX, 0),
      y: generateMockDistribution(selectedFeatureY, 0),
      z: generateMockDistribution(selectedFeatureZ, 0),
    };

    return { diabetic: diabeticPoints, nonDiabetic: nonDiabeticPoints };
  }, [selectedFeatureX, selectedFeatureY, selectedFeatureZ]);

  // Download chart handler
  const downloadChart = (chartId: string, filename: string) => {
    const element = document.getElementById(chartId);
    if (!element) return;

    // For Plotly charts, use Plotly's download feature
    // This is a simplified version - in production, use Plotly.downloadImage
    console.log(`Downloading chart: ${filename}`);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <ErrorMessage message={error} />
        <Button onClick={fetchData} className="mt-4" icon={<FiRefreshCw />}>
          Retry
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
            Data Visualization Dashboard
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Interactive visualizations and statistical analysis of diabetes dataset
          </p>
        </div>
        <Button onClick={fetchData} variant="outline" icon={<FiRefreshCw />}>
          Refresh
        </Button>
      </div>

      {/* Summary Cards */}
      {summaryStats && (
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <Card className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20">
            <div>
              <p className="text-sm font-medium text-blue-600 dark:text-blue-400">Total Patients</p>
              <p className="text-3xl font-bold text-blue-900 dark:text-blue-100 mt-1">
                {summaryStats.totalPatients.toLocaleString()}
              </p>
            </div>
          </Card>

          <Card className="bg-gradient-to-br from-danger-50 to-danger-100 dark:from-danger-900/20 dark:to-danger-800/20">
            <div>
              <p className="text-sm font-medium text-danger-600 dark:text-danger-400">
                Diabetic %
              </p>
              <p className="text-3xl font-bold text-danger-900 dark:text-danger-100 mt-1">
                {summaryStats.diabeticPercentage.toFixed(1)}%
              </p>
              <p className="text-xs text-danger-700 dark:text-danger-300 mt-1">
                {summaryStats.diabeticCount} patients
              </p>
            </div>
          </Card>

          <Card className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20">
            <div>
              <p className="text-sm font-medium text-purple-600 dark:text-purple-400">
                Average Age
              </p>
              <p className="text-3xl font-bold text-purple-900 dark:text-purple-100 mt-1">
                {summaryStats.avgAge.toFixed(1)}
              </p>
              <p className="text-xs text-purple-700 dark:text-purple-300 mt-1">years</p>
            </div>
          </Card>

          <Card className="bg-gradient-to-br from-warning-50 to-warning-100 dark:from-warning-900/20 dark:to-warning-800/20">
            <div>
              <p className="text-sm font-medium text-warning-600 dark:text-warning-400">
                Average BMI
              </p>
              <p className="text-3xl font-bold text-warning-900 dark:text-warning-100 mt-1">
                {summaryStats.avgBMI.toFixed(1)}
              </p>
              <p className="text-xs text-warning-700 dark:text-warning-300 mt-1">kg/m²</p>
            </div>
          </Card>

          <Card className="bg-gradient-to-br from-success-50 to-success-100 dark:from-success-900/20 dark:to-success-800/20">
            <div>
              <p className="text-sm font-medium text-success-600 dark:text-success-400">
                Features
              </p>
              <p className="text-3xl font-bold text-success-900 dark:text-success-100 mt-1">
                {dataStats?.features_count || 8}
              </p>
              <p className="text-xs text-success-700 dark:text-success-300 mt-1">dimensions</p>
            </div>
          </Card>
        </div>
      )}

      {/* Global Controls */}
      <Card>
        <div className="flex flex-wrap gap-4 items-center">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Color Scheme
            </label>
            <select
              value={colorScheme}
              onChange={(e) => setColorScheme(e.target.value as keyof typeof COLOR_SCHEMES)}
              className="input"
            >
              {Object.entries(COLOR_SCHEMES).map(([key, value]) => (
                <option key={key} value={key}>
                  {value.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Filter by Outcome
            </label>
            <select
              value={outcomeFilter}
              onChange={(e) =>
                setOutcomeFilter(e.target.value as 'all' | 'diabetic' | 'non-diabetic')
              }
              className="input"
            >
              <option value="all">All Patients</option>
              <option value="diabetic">Diabetic Only</option>
              <option value="non-diabetic">Non-Diabetic Only</option>
            </select>
          </div>
        </div>
      </Card>

      {/* Chart 1: Feature Distribution */}
      <Card>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                <FiBarChart2 className="w-5 h-5 text-primary-600" />
                Feature Distribution Analysis
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Compare distributions between diabetic and non-diabetic patients
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              icon={<FiDownload />}
              onClick={() => downloadChart('feature-dist', 'feature-distribution.png')}
            >
              Download
            </Button>
          </div>

          <div className="flex gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Select Feature
              </label>
              <select
                value={selectedFeature}
                onChange={(e) => setSelectedFeature(e.target.value)}
                className="input"
              >
                {FEATURES.map((feature) => (
                  <option key={feature} value={feature}>
                    {feature}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Histogram */}
            <div id="feature-dist">
              <Plot
                data={[
                  {
                    x: featureDistributionData.nonDiabetic,
                    type: 'histogram',
                    name: 'Non-Diabetic',
                    marker: { color: 'rgba(34, 197, 94, 0.7)' },
                    opacity: 0.7,
                  },
                  {
                    x: featureDistributionData.diabetic,
                    type: 'histogram',
                    name: 'Diabetic',
                    marker: { color: 'rgba(239, 68, 68, 0.7)' },
                    opacity: 0.7,
                  },
                ]}
                layout={{
                  title: `${selectedFeature} Distribution`,
                  xaxis: { title: selectedFeature },
                  yaxis: { title: 'Count' },
                  barmode: 'overlay',
                  height: 400,
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                }}
                config={{ responsive: true, displayModeBar: true }}
                style={{ width: '100%' }}
              />
            </div>

            {/* Box Plot */}
            <div>
              <Plot
                data={[
                  {
                    y: featureDistributionData.nonDiabetic,
                    type: 'box',
                    name: 'Non-Diabetic',
                    marker: { color: 'rgba(34, 197, 94, 0.7)' },
                  },
                  {
                    y: featureDistributionData.diabetic,
                    type: 'box',
                    name: 'Diabetic',
                    marker: { color: 'rgba(239, 68, 68, 0.7)' },
                  },
                ]}
                layout={{
                  title: `${selectedFeature} Box Plot`,
                  yaxis: { title: selectedFeature },
                  height: 400,
                  paper_bgcolor: 'rgba(0,0,0,0)',
                  plot_bgcolor: 'rgba(0,0,0,0)',
                }}
                config={{ responsive: true }}
                style={{ width: '100%' }}
              />
            </div>
          </div>

          {/* Summary Statistics */}
          {dataStats && dataStats.feature_statistics[selectedFeature] && (
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <div>
                <p className="text-xs text-gray-600 dark:text-gray-400">Mean</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {dataStats.feature_statistics[selectedFeature].mean.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-600 dark:text-gray-400">Std Dev</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {dataStats.feature_statistics[selectedFeature].std.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-600 dark:text-gray-400">Min</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {dataStats.feature_statistics[selectedFeature].min.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-600 dark:text-gray-400">Median</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {dataStats.feature_statistics[selectedFeature].median.toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-600 dark:text-gray-400">Max</p>
                <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {dataStats.feature_statistics[selectedFeature].max.toFixed(2)}
                </p>
              </div>
            </div>
          )}
        </div>
      </Card>

      {/* Chart 2: Correlation Heatmap */}
      <Card>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                <FiGrid className="w-5 h-5 text-primary-600" />
                Feature Correlation Heatmap
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Correlation coefficients between all features
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              icon={<FiDownload />}
              onClick={() => downloadChart('correlation-heatmap', 'correlation-heatmap.png')}
            >
              Download
            </Button>
          </div>

          <div id="correlation-heatmap">
            <Plot
              data={[
                {
                  z: correlationMatrix,
                  x: FEATURES,
                  y: FEATURES,
                  type: 'heatmap',
                  colorscale: COLOR_SCHEMES[colorScheme].colors.map((color, i, arr) => [
                    i / (arr.length - 1),
                    color,
                  ]),
                  hoverongaps: false,
                  hovertemplate: '<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>',
                },
              ]}
              layout={{
                title: 'Feature Correlation Matrix',
                xaxis: { side: 'bottom' },
                yaxis: { autorange: 'reversed' },
                height: 600,
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
              }}
              config={{ responsive: true }}
              style={{ width: '100%' }}
            />
          </div>
        </div>
      </Card>

      {/* Chart 3: 3D Scatter Plot */}
      <Card>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                <FiLayers className="w-5 h-5 text-primary-600" />
                3D Feature Space Visualization
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Interactive 3D scatter plot - rotate and zoom to explore
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              icon={<FiDownload />}
              onClick={() => downloadChart('scatter-3d', '3d-scatter.png')}
            >
              Download
            </Button>
          </div>

          <div className="flex gap-4 flex-wrap">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                X Axis
              </label>
              <select
                value={selectedFeatureX}
                onChange={(e) => setSelectedFeatureX(e.target.value)}
                className="input"
              >
                {FEATURES.map((feature) => (
                  <option key={feature} value={feature}>
                    {feature}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Y Axis
              </label>
              <select
                value={selectedFeatureY}
                onChange={(e) => setSelectedFeatureY(e.target.value)}
                className="input"
              >
                {FEATURES.map((feature) => (
                  <option key={feature} value={feature}>
                    {feature}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Z Axis
              </label>
              <select
                value={selectedFeatureZ}
                onChange={(e) => setSelectedFeatureZ(e.target.value)}
                className="input"
              >
                {FEATURES.map((feature) => (
                  <option key={feature} value={feature}>
                    {feature}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div id="scatter-3d">
            <Plot
              data={[
                {
                  x: scatter3DData.nonDiabetic.x,
                  y: scatter3DData.nonDiabetic.y,
                  z: scatter3DData.nonDiabetic.z,
                  mode: 'markers',
                  type: 'scatter3d',
                  name: 'Non-Diabetic',
                  marker: {
                    size: 5,
                    color: 'rgba(34, 197, 94, 0.8)',
                    line: {
                      color: 'rgba(34, 197, 94, 1)',
                      width: 0.5,
                    },
                  },
                },
                {
                  x: scatter3DData.diabetic.x,
                  y: scatter3DData.diabetic.y,
                  z: scatter3DData.diabetic.z,
                  mode: 'markers',
                  type: 'scatter3d',
                  name: 'Diabetic',
                  marker: {
                    size: 5,
                    color: 'rgba(239, 68, 68, 0.8)',
                    line: {
                      color: 'rgba(239, 68, 68, 1)',
                      width: 0.5,
                    },
                  },
                },
              ]}
              layout={{
                title: '3D Feature Space',
                scene: {
                  xaxis: { title: selectedFeatureX },
                  yaxis: { title: selectedFeatureY },
                  zaxis: { title: selectedFeatureZ },
                },
                height: 600,
                paper_bgcolor: 'rgba(0,0,0,0)',
              }}
              config={{ responsive: true }}
              style={{ width: '100%' }}
            />
          </div>
        </div>
      </Card>

      {/* Chart 4: Pairplot Matrix */}
      <Card>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                <FiActivity className="w-5 h-5 text-primary-600" />
                Pairplot Matrix
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Pairwise relationships between selected features
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              icon={<FiDownload />}
              onClick={() => downloadChart('pairplot', 'pairplot.png')}
            >
              Download
            </Button>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Select Features for Pairplot (3 features)
            </label>
            <div className="flex gap-4 flex-wrap">
              {[0, 1, 2].map((index) => (
                <select
                  key={index}
                  value={selectedPairFeatures[index]}
                  onChange={(e) => {
                    const newFeatures = [...selectedPairFeatures];
                    newFeatures[index] = e.target.value;
                    setSelectedPairFeatures(newFeatures);
                  }}
                  className="input"
                >
                  {FEATURES.map((feature) => (
                    <option key={feature} value={feature}>
                      {feature}
                    </option>
                  ))}
                </select>
              ))}
            </div>
          </div>

          <div id="pairplot" className="grid grid-cols-3 gap-2">
            {selectedPairFeatures.map((featureY, i) =>
              selectedPairFeatures.map((featureX, j) => {
                const key = `${i}-${j}`;

                // Diagonal: show distribution
                if (i === j) {
                  return (
                    <div key={key} className="border border-gray-200 dark:border-gray-700 rounded">
                      <Plot
                        data={[
                          {
                            x: generateMockDistribution(featureX, 0),
                            type: 'histogram',
                            name: 'Non-Diabetic',
                            marker: { color: 'rgba(34, 197, 94, 0.7)' },
                          },
                        ]}
                        layout={{
                          title: { text: featureX, font: { size: 10 } },
                          showlegend: false,
                          height: 200,
                          margin: { l: 40, r: 20, t: 30, b: 40 },
                          paper_bgcolor: 'rgba(0,0,0,0)',
                          plot_bgcolor: 'rgba(0,0,0,0)',
                        }}
                        config={{ displayModeBar: false }}
                        style={{ width: '100%' }}
                      />
                    </div>
                  );
                }

                // Off-diagonal: scatter plot
                return (
                  <div key={key} className="border border-gray-200 dark:border-gray-700 rounded">
                    <Plot
                      data={[
                        {
                          x: generateMockDistribution(featureX, 0),
                          y: generateMockDistribution(featureY, 0),
                          mode: 'markers',
                          type: 'scatter',
                          name: 'Non-Diabetic',
                          marker: { size: 3, color: 'rgba(34, 197, 94, 0.5)' },
                        },
                        {
                          x: generateMockDistribution(featureX, 1),
                          y: generateMockDistribution(featureY, 1),
                          mode: 'markers',
                          type: 'scatter',
                          name: 'Diabetic',
                          marker: { size: 3, color: 'rgba(239, 68, 68, 0.5)' },
                        },
                      ]}
                      layout={{
                        showlegend: false,
                        height: 200,
                        margin: { l: 40, r: 20, t: 20, b: 40 },
                        xaxis: { title: j === 0 ? featureX : '', titlefont: { size: 9 } },
                        yaxis: { title: i === selectedPairFeatures.length - 1 ? featureY : '', titlefont: { size: 9 } },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                      }}
                      config={{ displayModeBar: false }}
                      style={{ width: '100%' }}
                    />
                  </div>
                );
              })
            )}
          </div>
        </div>
      </Card>

      {/* Info Card */}
      <Card className="bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800">
        <div className="flex items-start gap-3">
          <div className="w-10 h-10 rounded-lg bg-blue-100 dark:bg-blue-900/40 flex items-center justify-center flex-shrink-0">
            <FiBarChart2 className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          </div>
          <div>
            <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
              Visualization Tips
            </h4>
            <ul className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
              <li>" Hover over charts to see detailed values and statistics</li>
              <li>" Use the 3D scatter plot controls to rotate and zoom the view</li>
              <li>" Click on legend items to toggle visibility of data series</li>
              <li>" Correlation values range from -1 (negative) to +1 (positive correlation)</li>
              <li>
                " The pairplot matrix shows relationships between multiple features simultaneously
              </li>
              <li>" Download any chart as a PNG image for use in reports</li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default VisualizationDashboard;
