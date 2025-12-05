import React, { useEffect, useRef, useState } from 'react';
import {
  FiCheckCircle,
  FiAlertTriangle,
  FiAlertCircle,
  FiActivity,
  FiUsers,
  FiTrendingUp,
  FiTrendingDown,
  FiClock,
  FiDownload,
  FiMail,
  FiSave,
  FiHeart,
  FiDroplet,
  FiZap,
  FiAward,
} from 'react-icons/fi';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import { Chart as ChartJS, ArcElement, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { Doughnut, Bar } from 'react-chartjs-2';
import Plot from 'react-plotly.js';
import { Card, Button } from '@/components/common';
import type { ComprehensivePredictionOutput } from '@/types/api';

// Register Chart.js components
ChartJS.register(ArcElement, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

interface PredictionResultsProps {
  result: ComprehensivePredictionOutput;
}

const PredictionResults: React.FC<PredictionResultsProps> = ({ result }) => {
  const [isVisible, setIsVisible] = useState(false);
  const resultsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Trigger animation on mount
    setIsVisible(true);

    // Register Chart.js plugins for better tooltips
    ChartJS.defaults.plugins.tooltip.backgroundColor = 'rgba(0, 0, 0, 0.8)';
    ChartJS.defaults.plugins.tooltip.padding = 12;
    ChartJS.defaults.plugins.tooltip.cornerRadius = 8;
  }, []);

  // Get risk color and icon
  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel.toLowerCase()) {
      case 'low':
        return {
          bg: 'bg-success-50 dark:bg-success-900/20',
          border: 'border-success-200 dark:border-success-800',
          text: 'text-success-700 dark:text-success-300',
          icon: <FiCheckCircle className="w-12 h-12" />,
          chartColor: '#22c55e',
          gradientFrom: '#dcfce7',
          gradientTo: '#22c55e',
        };
      case 'medium':
        return {
          bg: 'bg-warning-50 dark:bg-warning-900/20',
          border: 'border-warning-200 dark:border-warning-800',
          text: 'text-warning-700 dark:text-warning-300',
          icon: <FiAlertTriangle className="w-12 h-12" />,
          chartColor: '#f59e0b',
          gradientFrom: '#fef3c7',
          gradientTo: '#f59e0b',
        };
      case 'high':
        return {
          bg: 'bg-danger-50 dark:bg-danger-900/20',
          border: 'border-danger-200 dark:border-danger-800',
          text: 'text-danger-700 dark:text-danger-300',
          icon: <FiAlertCircle className="w-12 h-12" />,
          chartColor: '#ef4444',
          gradientFrom: '#fee2e2',
          gradientTo: '#ef4444',
        };
      default:
        return {
          bg: 'bg-gray-50 dark:bg-gray-900/20',
          border: 'border-gray-200 dark:border-gray-800',
          text: 'text-gray-700 dark:text-gray-300',
          icon: <FiActivity className="w-12 h-12" />,
          chartColor: '#6b7280',
          gradientFrom: '#e5e7eb',
          gradientTo: '#6b7280',
        };
    }
  };

  const riskStyle = getRiskColor(result.risk_level);

  // Export functionality
  // Export functionality
  const handleExportPDF = async () => {
    if (!resultsRef.current) return;

    try {
      const canvas = await html2canvas(resultsRef.current, {
        scale: 2, // Better quality
        useCORS: true,
        logging: false,
        backgroundColor: '#ffffff', // Ensure white background
      });

      const imgData = canvas.toDataURL('image/png');
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4',
      });

      const imgWidth = 210; // A4 width in mm
      const pageHeight = 297; // A4 height in mm
      const imgHeight = (canvas.height * imgWidth) / canvas.width;

      let heightLeft = imgHeight;
      let position = 0;

      // Add first page
      pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
      heightLeft -= pageHeight;

      // Add subsequent pages if content is long
      while (heightLeft >= 0) {
        position = heightLeft - imgHeight;
        pdf.addPage();
        pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
        heightLeft -= pageHeight;
      }

      pdf.save(`diabetes-prediction-results-${new Date().toISOString().split('T')[0]}.pdf`);
    } catch (error) {
      console.error('Error exporting PDF:', error);
      alert('Failed to export PDF. Please try again.');
    }
  };

  const handleSendEmail = () => {
    const subject = encodeURIComponent('Diabetes Risk Assessment Results');
    const body = encodeURIComponent(
      `Diabetes Risk Prediction Results\n\n` +
      `Risk Level: ${result.risk_level}\n` +
      `Probability: ${(result.ensemble_probability * 100).toFixed(1)}%\n` +
      `Prediction: ${result.ensemble_label}\n\n` +
      `Please see attached detailed report for full analysis.`
    );
    window.location.href = `mailto:?subject=${subject}&body=${body}`;
  };

  const handleSaveLocal = () => {
    const dataStr = JSON.stringify(result, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `diabetes-prediction-${new Date().toISOString()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Circular progress chart data
  const circularProgressData = {
    labels: ['Risk', 'Safe'],
    datasets: [
      {
        data: [result.ensemble_probability * 100, (1 - result.ensemble_probability) * 100],
        backgroundColor: [riskStyle.chartColor, '#e5e7eb'],
        borderWidth: 0,
        cutout: '75%',
      },
    ],
  };

  const circularProgressOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function (context: any) {
            return `${context.label}: ${context.parsed.toFixed(1)}%`;
          },
        },
      },
    },
  };

  // Feature importance bar chart
  const featureImportanceData = result.shap_explanation
    ? {
      labels: result.shap_explanation.feature_contributions
        .slice(0, 10)
        .map((c) => c.feature),
      datasets: [
        {
          label: 'Feature Importance',
          data: result.shap_explanation.feature_contributions
            .slice(0, 10)
            .map((c) => Math.abs(c.contribution)),
          backgroundColor: result.shap_explanation.feature_contributions
            .slice(0, 10)
            .map((c) => (c.contribution > 0 ? 'rgba(239, 68, 68, 0.7)' : 'rgba(34, 197, 94, 0.7)')),
          borderColor: result.shap_explanation.feature_contributions
            .slice(0, 10)
            .map((c) => (c.contribution > 0 ? 'rgb(239, 68, 68)' : 'rgb(34, 197, 94)')),
          borderWidth: 2,
        },
      ],
    }
    : null;

  const featureImportanceOptions = {
    indexAxis: 'y' as const,
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function (context: any) {
            const contribution = result.shap_explanation?.feature_contributions[context.dataIndex];
            return `Impact: ${contribution?.contribution.toFixed(3)} (${contribution?.impact})`;
          },
        },
      },
    },
    scales: {
      x: {
        beginAtZero: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
      },
      y: {
        grid: {
          display: false,
        },
      },
    },
  };

  // SHAP Waterfall chart using Plotly
  const getWaterfallData = () => {
    if (!result.shap_explanation) return null;

    const contributions = result.shap_explanation.feature_contributions.slice(0, 10);
    const labels = ['Base Value', ...contributions.map((c) => c.feature), 'Final Prediction'];

    let runningTotal = result.shap_explanation.base_value;
    const values = [runningTotal];
    const measures = ['absolute'];

    contributions.forEach((c) => {
      values.push(c.contribution);
      measures.push('relative');
      runningTotal += c.contribution;
    });

    values.push(runningTotal);
    measures.push('total');

    return {
      x: labels,
      y: values,
      type: 'waterfall' as const,
      orientation: 'v' as const,
      measure: measures,
      connector: {
        line: {
          color: 'rgba(107, 114, 128, 0.5)',
          width: 2,
        },
      },
      increasing: {
        marker: {
          color: 'rgba(239, 68, 68, 0.7)',
        },
      },
      decreasing: {
        marker: {
          color: 'rgba(34, 197, 94, 0.7)',
        },
      },
      totals: {
        marker: {
          color: 'rgba(59, 130, 246, 0.7)',
        },
      },
      text: values.map((v) => v.toFixed(3)),
      textposition: 'outside' as const,
    };
  };

  const waterfallData = getWaterfallData();

  // Get recommendation icon
  const getRecommendationIcon = (category: string) => {
    const lowerCategory = category.toLowerCase();
    if (lowerCategory.includes('weight') || lowerCategory.includes('bmi')) return <FiActivity />;
    if (lowerCategory.includes('glucose') || lowerCategory.includes('blood')) return <FiDroplet />;
    if (lowerCategory.includes('pressure')) return <FiHeart />;
    if (lowerCategory.includes('activity') || lowerCategory.includes('exercise')) return <FiZap />;
    return <FiAward />;
  };

  return (
    <div
      ref={resultsRef}
      className={`space-y-6 transition-all duration-700 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
        }`}
    >
      {/* Export Actions Bar */}
      <div className="flex flex-wrap gap-3 justify-end">
        <Button
          variant="outline"
          size="sm"
          onClick={handleExportPDF}
          icon={<FiDownload />}
        >
          Export PDF
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={handleSendEmail}
          icon={<FiMail />}
        >
          Email Results
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={handleSaveLocal}
          icon={<FiSave />}
        >
          Save Locally
        </Button>
      </div>

      {/* Section 1: Risk Assessment Card with Circular Progress */}
      <Card className={`${riskStyle.bg} ${riskStyle.border} border-2 overflow-hidden`}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Left: Text Information */}
          <div className="flex flex-col justify-center">
            <div className={`flex items-center gap-3 mb-4 ${riskStyle.text}`}>
              {riskStyle.icon}
              <div>
                <h2 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
                  {result.ensemble_label}
                </h2>
                <p className="text-xl font-semibold">
                  {result.risk_level} Risk
                </p>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600 dark:text-gray-400">Probability:</span>
                <span className={`text-2xl font-bold ${riskStyle.text}`}>
                  {(result.ensemble_probability * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600 dark:text-gray-400">Confidence:</span>
                <span className="text-lg font-semibold text-gray-700 dark:text-gray-300">
                  {(result.ensemble_confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center gap-2 pt-3 border-t border-gray-200 dark:border-gray-700">
                <FiClock className="w-4 h-4 text-gray-500" />
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  Processed in {result.processing_time_ms.toFixed(0)}ms
                </span>
              </div>
            </div>
          </div>

          {/* Right: Circular Progress Chart */}
          <div className="flex items-center justify-center">
            <div className="relative w-64 h-64">
              <Doughnut data={circularProgressData} options={circularProgressOptions} />
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <div className={`text-4xl font-bold ${riskStyle.text}`}>
                    {(result.ensemble_probability * 100).toFixed(0)}%
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    Risk Score
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Section 2: Model Predictions Comparison Table */}
      <Card title="Model Predictions Comparison" subtitle="Individual and ensemble predictions">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="text-left py-3 px-4 font-semibold text-gray-700 dark:text-gray-300">
                  Model
                </th>
                <th className="text-center py-3 px-4 font-semibold text-gray-700 dark:text-gray-300">
                  Prediction
                </th>
                <th className="text-center py-3 px-4 font-semibold text-gray-700 dark:text-gray-300">
                  Probability
                </th>
                <th className="text-center py-3 px-4 font-semibold text-gray-700 dark:text-gray-300">
                  Confidence
                </th>
                <th className="text-left py-3 px-4 font-semibold text-gray-700 dark:text-gray-300">
                  Visual
                </th>
              </tr>
            </thead>
            <tbody>
              {result.model_predictions.map((model, idx) => {
                const isConsensus = model.prediction === result.ensemble_prediction;
                return (
                  <tr
                    key={model.model_name}
                    className={`border-b border-gray-100 dark:border-gray-800 ${isConsensus ? 'bg-primary-50/30 dark:bg-primary-900/10' : ''
                      }`}
                  >
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <span className="font-medium capitalize">
                          {model.model_name.replace(/_/g, ' ')}
                        </span>
                        {isConsensus && (
                          <FiCheckCircle className="w-4 h-4 text-primary-600" title="Agrees with ensemble" />
                        )}
                      </div>
                    </td>
                    <td className="py-3 px-4 text-center">
                      <span
                        className={`px-3 py-1 rounded-full text-sm font-semibold ${model.prediction === 1
                            ? 'bg-danger-100 text-danger-800 dark:bg-danger-800 dark:text-danger-100'
                            : 'bg-success-100 text-success-800 dark:bg-success-800 dark:text-success-100'
                          }`}
                      >
                        {model.prediction_label}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-center font-semibold">
                      {(model.probability * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-4 text-center">
                      {(model.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-4">
                      <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className={`h-full rounded-full ${model.probability < 0.3
                              ? 'bg-success-500'
                              : model.probability < 0.7
                                ? 'bg-warning-500'
                                : 'bg-danger-500'
                            }`}
                          style={{ width: `${model.probability * 100}%` }}
                        />
                      </div>
                    </td>
                  </tr>
                );
              })}
              {/* Ensemble Row */}
              <tr className="bg-primary-100 dark:bg-primary-900/30 font-bold">
                <td className="py-3 px-4">
                  <div className="flex items-center gap-2">
                    <FiAward className="w-5 h-5 text-primary-600" />
                    <span>Ensemble</span>
                  </div>
                </td>
                <td className="py-3 px-4 text-center">
                  <span
                    className={`px-3 py-1 rounded-full text-sm font-bold ${result.ensemble_prediction === 1
                        ? 'bg-danger-200 text-danger-900 dark:bg-danger-700 dark:text-danger-100'
                        : 'bg-success-200 text-success-900 dark:bg-success-700 dark:text-success-100'
                      }`}
                  >
                    {result.ensemble_label}
                  </span>
                </td>
                <td className="py-3 px-4 text-center">
                  {(result.ensemble_probability * 100).toFixed(1)}%
                </td>
                <td className="py-3 px-4 text-center">
                  {(result.ensemble_confidence * 100).toFixed(1)}%
                </td>
                <td className="py-3 px-4">
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className={`h-full rounded-full ${result.ensemble_probability < 0.3
                          ? 'bg-success-600'
                          : result.ensemble_probability < 0.7
                            ? 'bg-warning-600'
                            : 'bg-danger-600'
                        }`}
                      style={{ width: `${result.ensemble_probability * 100}%` }}
                    />
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Model Agreement Summary */}
        <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
          <p className="text-sm text-gray-600 dark:text-gray-400">
            <strong>Model Agreement:</strong>{' '}
            {result.model_predictions.filter((m) => m.prediction === result.ensemble_prediction).length}/
            {result.model_predictions.length} models agree with the ensemble prediction
          </p>
        </div>
      </Card>

      {/* Section 3: Feature Importance Bar Chart */}
      {featureImportanceData && (
        <Card
          title="Feature Importance"
          subtitle="Top 10 features ranked by contribution to prediction"
        >
          <div className="h-96">
            <Bar data={featureImportanceData} options={featureImportanceOptions} />
          </div>
          <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
            <div className="flex items-center gap-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-danger-500 rounded" />
                <span className="text-gray-600 dark:text-gray-400">Increases Risk</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 bg-success-500 rounded" />
                <span className="text-gray-600 dark:text-gray-400">Decreases Risk</span>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Section 4: SHAP Waterfall Chart */}
      {waterfallData && result.shap_available && (
        <Card
          title="SHAP Explanation Waterfall"
          subtitle="How each feature moves the prediction from base value to final result"
        >
          <div className="h-96">
            <Plot
              data={[waterfallData]}
              layout={{
                autosize: true,
                showlegend: false,
                xaxis: {
                  title: 'Features',
                  tickangle: -45,
                },
                yaxis: {
                  title: 'Contribution to Prediction',
                },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {
                  color: '#6b7280',
                },
                margin: {
                  l: 60,
                  r: 40,
                  t: 40,
                  b: 100,
                },
              }}
              config={{ responsive: true, displayModeBar: false }}
              style={{ width: '100%', height: '100%' }}
            />
          </div>
          <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              <strong>How to read:</strong> The chart shows how the prediction moves from the base
              value (average prediction) to the final prediction. Red bars push the prediction
              toward diabetes, while green bars push it away.
            </p>
          </div>
        </Card>
      )}

      {/* Risk Factors Grid */}
      {result.risk_factors.length > 0 && (
        <Card title="Identified Risk Factors" subtitle="Areas of concern based on your input">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {result.risk_factors.map((factor, idx) => {
              const riskColor =
                factor.risk_level.toLowerCase() === 'high'
                  ? 'border-danger-500 bg-danger-50 dark:bg-danger-900/20'
                  : 'border-warning-500 bg-warning-50 dark:bg-warning-900/20';

              return (
                <div
                  key={idx}
                  className={`p-4 border-2 rounded-lg ${riskColor} transform transition-all duration-300 hover:scale-105`}
                >
                  <div className="flex items-start justify-between mb-2">
                    <h4 className="font-semibold text-sm text-gray-900 dark:text-gray-100">
                      {factor.factor}
                    </h4>
                    <span
                      className={`px-2 py-1 text-xs font-semibold rounded ${factor.risk_level.toLowerCase() === 'high'
                          ? 'bg-danger-100 text-danger-800 dark:bg-danger-800 dark:text-danger-100'
                          : 'bg-warning-100 text-warning-800 dark:bg-warning-800 dark:text-warning-100'
                        }`}
                    >
                      {factor.risk_level} Risk
                    </span>
                  </div>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    Current Value: <strong>{factor.current_value.toFixed(2)}</strong>
                  </p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    {factor.is_modifiable ? (
                      <span className="text-success-600 dark:text-success-400">
                        ✓ Modifiable through lifestyle changes
                      </span>
                    ) : (
                      <span className="text-gray-500">Non-modifiable factor</span>
                    )}
                  </p>
                </div>
              );
            })}
          </div>
        </Card>
      )}

      {/* Section 5: Personalized Recommendations */}
      {result.recommendations.length > 0 && (
        <Card
          title="Personalized Recommendations"
          subtitle="Evidence-based action items to reduce your diabetes risk"
        >
          <div className="space-y-4">
            {result.recommendations.map((rec, idx) => {
              const priorityColor =
                rec.priority === 'High'
                  ? 'border-l-danger-500'
                  : rec.priority === 'Medium'
                    ? 'border-l-warning-500'
                    : 'border-l-primary-500';

              return (
                <div
                  key={idx}
                  className={`p-4 border-l-4 ${priorityColor} bg-gray-50 dark:bg-gray-900/50 rounded-r-lg transform transition-all duration-300 hover:shadow-md`}
                  style={{ animationDelay: `${idx * 100}ms` }}
                >
                  <div className="flex items-start gap-3">
                    <div className={`mt-1 ${rec.priority === 'High' ? 'text-danger-600' :
                        rec.priority === 'Medium' ? 'text-warning-600' :
                          'text-primary-600'
                      }`}>
                      {getRecommendationIcon(rec.category)}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-start justify-between mb-2">
                        <h4 className="font-semibold text-gray-900 dark:text-gray-100">
                          {rec.category}
                        </h4>
                        <span
                          className={`px-2 py-1 text-xs font-semibold rounded ${rec.priority === 'High'
                              ? 'bg-danger-100 text-danger-800 dark:bg-danger-800 dark:text-danger-100'
                              : rec.priority === 'Medium'
                                ? 'bg-warning-100 text-warning-800 dark:bg-warning-800 dark:text-warning-100'
                                : 'bg-primary-100 text-primary-800 dark:bg-primary-800 dark:text-primary-100'
                            }`}
                        >
                          {rec.priority} Priority
                        </span>
                      </div>
                      <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                        {rec.recommendation}
                      </p>
                      <p className="text-xs text-gray-600 dark:text-gray-400 italic">
                        <strong>Why:</strong> {rec.rationale}
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Resources Section */}
          <div className="mt-6 p-4 bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 rounded-lg">
            <h5 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">
              Additional Resources
            </h5>
            <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
              <li>• American Diabetes Association: <a href="https://diabetes.org" className="text-primary-600 hover:underline">diabetes.org</a></li>
              <li>• CDC Diabetes Prevention Program: <a href="https://cdc.gov/diabetes" className="text-primary-600 hover:underline">cdc.gov/diabetes</a></li>
              <li>• Consult with your healthcare provider for personalized medical advice</li>
            </ul>
          </div>
        </Card>
      )}

      {/* Similar Patients */}
      {result.similar_patients.length > 0 && (
        <Card title="Similar Patient Outcomes" subtitle="Historical cases with similar profiles">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {result.similar_patients.map((patient, idx) => (
              <div
                key={idx}
                className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:shadow-lg transition-shadow"
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <FiUsers className="w-5 h-5 text-primary-600" />
                    <span className="text-sm font-semibold">Case {idx + 1}</span>
                  </div>
                  <span
                    className={`px-2 py-1 text-xs font-semibold rounded ${patient.outcome === 1
                        ? 'bg-danger-100 text-danger-800 dark:bg-danger-800 dark:text-danger-100'
                        : 'bg-success-100 text-success-800 dark:bg-success-800 dark:text-success-100'
                      }`}
                  >
                    {patient.outcome_label}
                  </span>
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  <p className="mb-2">
                    Similarity: <strong>{(patient.similarity_score * 100).toFixed(1)}%</strong>
                  </p>
                  {patient.key_similarities.length > 0 && (
                    <div>
                      <p className="text-xs mb-1">Matching features:</p>
                      <div className="flex flex-wrap gap-1">
                        {patient.key_similarities.map((feature) => (
                          <span
                            key={feature}
                            className="px-2 py-1 text-xs bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded"
                          >
                            {feature}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};

export default PredictionResults;
