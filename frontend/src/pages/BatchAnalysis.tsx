import React, { useState, useMemo, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import Papa from 'papaparse';
import {
  FiUpload,
  FiX,
  FiCheckCircle,
  FiAlertCircle,
  FiDownload,
  FiSearch,
  FiFilter,
  FiChevronUp,
  FiChevronDown,
  FiEye,
  FiFileText,
  FiBarChart2,
} from 'react-icons/fi';
import { Pie, Bar } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement } from 'chart.js';
import { Button, Card, LoadingSpinner, ErrorMessage } from '@/components/common';
import PredictionResults from '@/components/PredictionResults';
import type { PatientInput, PredictionOutput, ComprehensivePredictionOutput } from '@/types/api';
import { predictionApi, handleApiError } from '@/services/api';

// Register ChartJS components
ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement);

// Expected CSV columns
const EXPECTED_COLUMNS = [
  'Pregnancies',
  'Glucose',
  'BloodPressure',
  'SkinThickness',
  'Insulin',
  'BMI',
  'DiabetesPedigreeFunction',
  'Age',
];

// Patient with prediction result
interface PatientWithPrediction extends PatientInput {
  id: string;
  prediction?: PredictionOutput;
  detailedPrediction?: ComprehensivePredictionOutput;
}

type SortColumn = keyof PatientInput | 'prediction' | 'probability' | 'risk_level';
type SortDirection = 'asc' | 'desc';

const BatchAnalysis: React.FC = () => {
  // File upload state
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [csvData, setCsvData] = useState<PatientWithPrediction[]>([]);
  const [csvError, setCsvError] = useState<string | null>(null);
  const [showPreview, setShowPreview] = useState(false);

  // Processing state
  const [processing, setProcessing] = useState(false);
  const [processProgress, setProcessProgress] = useState(0);
  const [processedData, setProcessedData] = useState<PatientWithPrediction[]>([]);

  // Table state
  const [searchTerm, setSearchTerm] = useState('');
  const [sortColumn, setSortColumn] = useState<SortColumn>('probability');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [riskFilter, setRiskFilter] = useState<string>('all');

  // Detailed view state
  const [selectedPatient, setSelectedPatient] = useState<PatientWithPrediction | null>(null);
  const [loadingDetails, setLoadingDetails] = useState(false);

  // File drop handler
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setCsvFile(file);
    setCsvError(null);
    setProcessedData([]);

    // Parse CSV
    Papa.parse<Record<string, string>>(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        // Validate columns
        const headers = results.meta.fields || [];
        const missingColumns = EXPECTED_COLUMNS.filter((col) => !headers.includes(col));

        if (missingColumns.length > 0) {
          setCsvError(
            `Missing required columns: ${missingColumns.join(', ')}. Expected columns: ${EXPECTED_COLUMNS.join(', ')}`
          );
          return;
        }

        // Parse data
        const patients: PatientWithPrediction[] = results.data.map((row, index) => ({
          id: `patient-${index + 1}`,
          Pregnancies: parseFloat(row.Pregnancies) || 0,
          Glucose: parseFloat(row.Glucose) || 0,
          BloodPressure: parseFloat(row.BloodPressure) || 0,
          SkinThickness: parseFloat(row.SkinThickness) || 0,
          Insulin: parseFloat(row.Insulin) || 0,
          BMI: parseFloat(row.BMI) || 0,
          DiabetesPedigreeFunction: parseFloat(row.DiabetesPedigreeFunction) || 0,
          Age: parseFloat(row.Age) || 0,
        }));

        setCsvData(patients);
        setShowPreview(true);
      },
      error: (error) => {
        setCsvError(`Failed to parse CSV: ${error.message}`);
      },
    });
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
    },
    multiple: false,
  });

  // Process batch
  const processBatch = async () => {
    if (csvData.length === 0) return;

    setProcessing(true);
    setProcessProgress(0);

    try {
      // Process in batches to show progress
      const batchSize = 10;
      const results: PatientWithPrediction[] = [];

      for (let i = 0; i < csvData.length; i += batchSize) {
        const batch = csvData.slice(i, i + batchSize);
        const batchPatients = batch.map(({ id, ...patient }) => patient);

        const batchResult = await predictionApi.predictBatch({
          patients: batchPatients,
        });

        // Merge predictions with patient data
        batch.forEach((patient, index) => {
          results.push({
            ...patient,
            prediction: batchResult.predictions[index],
          });
        });

        setProcessProgress(Math.min(((i + batchSize) / csvData.length) * 100, 100));
      }

      setProcessedData(results);
      setShowPreview(false);
    } catch (error) {
      setCsvError(handleApiError(error));
    } finally {
      setProcessing(false);
    }
  };

  // Clear all
  const clearAll = () => {
    setCsvFile(null);
    setCsvData([]);
    setCsvError(null);
    setShowPreview(false);
    setProcessedData([]);
    setSearchTerm('');
    setSortColumn('probability');
    setSortDirection('desc');
    setRiskFilter('all');
  };

  // Sort and filter data
  const filteredAndSortedData = useMemo(() => {
    let data = [...processedData];

    // Filter by risk level
    if (riskFilter !== 'all') {
      data = data.filter((p) => p.prediction?.risk_level.toLowerCase() === riskFilter);
    }

    // Search filter
    if (searchTerm) {
      data = data.filter(
        (p) =>
          p.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
          p.prediction?.risk_level.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Sort
    data.sort((a, b) => {
      let aVal: any;
      let bVal: any;

      if (sortColumn === 'prediction') {
        aVal = a.prediction?.prediction || 0;
        bVal = b.prediction?.prediction || 0;
      } else if (sortColumn === 'probability') {
        aVal = a.prediction?.probability || 0;
        bVal = b.prediction?.probability || 0;
      } else if (sortColumn === 'risk_level') {
        const riskOrder: Record<string, number> = { low: 1, medium: 2, high: 3 };
        aVal = riskOrder[a.prediction?.risk_level.toLowerCase() || 'medium'];
        bVal = riskOrder[b.prediction?.risk_level.toLowerCase() || 'medium'];
      } else {
        aVal = a[sortColumn as keyof PatientInput];
        bVal = b[sortColumn as keyof PatientInput];
      }

      if (sortDirection === 'asc') {
        return aVal > bVal ? 1 : -1;
      } else {
        return aVal < bVal ? 1 : -1;
      }
    });

    return data;
  }, [processedData, searchTerm, sortColumn, sortDirection, riskFilter]);

  // Calculate statistics
  const statistics = useMemo(() => {
    if (processedData.length === 0) return null;

    const total = processedData.length;
    const lowRisk = processedData.filter((p) => p.prediction?.risk_level.toLowerCase() === 'low').length;
    const mediumRisk = processedData.filter((p) => p.prediction?.risk_level.toLowerCase() === 'medium').length;
    const highRisk = processedData.filter((p) => p.prediction?.risk_level.toLowerCase() === 'high').length;

    const avgProbability =
      processedData.reduce((sum, p) => sum + (p.prediction?.probability || 0), 0) / total;

    const highestRisk = [...processedData].sort(
      (a, b) => (b.prediction?.probability || 0) - (a.prediction?.probability || 0)
    );

    return {
      total,
      lowRisk,
      mediumRisk,
      highRisk,
      lowRiskPct: (lowRisk / total) * 100,
      mediumRiskPct: (mediumRisk / total) * 100,
      highRiskPct: (highRisk / total) * 100,
      avgProbability,
      highestRiskPatients: highestRisk.slice(0, 5),
    };
  }, [processedData]);

  // Handle sort
  const handleSort = (column: SortColumn) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('desc');
    }
  };

  // View patient details
  const viewPatientDetails = async (patient: PatientWithPrediction) => {
    setSelectedPatient(patient);

    // If we don't have detailed prediction, fetch it
    if (!patient.detailedPrediction) {
      setLoadingDetails(true);
      try {
        const { id, prediction, detailedPrediction, ...patientData } = patient;
        const detailed = await predictionApi.predictExplain(patientData);

        // Update the patient with detailed prediction
        const updatedData = processedData.map((p) =>
          p.id === patient.id ? { ...p, detailedPrediction: detailed } : p
        );
        setProcessedData(updatedData);
        setSelectedPatient({ ...patient, detailedPrediction: detailed });
      } catch (error) {
        console.error('Error fetching detailed prediction:', error);
      } finally {
        setLoadingDetails(false);
      }
    }
  };

  // Export results as CSV
  const exportResultsCSV = () => {
    if (processedData.length === 0) return;

    const headers = [
      ...EXPECTED_COLUMNS,
      'Prediction',
      'Probability',
      'Risk Level',
      'Confidence',
    ];

    const rows = processedData.map((p) => [
      p.Pregnancies,
      p.Glucose,
      p.BloodPressure,
      p.SkinThickness,
      p.Insulin,
      p.BMI,
      p.DiabetesPedigreeFunction,
      p.Age,
      p.prediction?.prediction_label || '',
      p.prediction?.probability.toFixed(4) || '',
      p.prediction?.risk_level || '',
      p.prediction?.confidence.toFixed(4) || '',
    ]);

    const csv = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `batch-predictions-${new Date().toISOString()}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Export summary report
  const exportSummaryReport = () => {
    if (!statistics) return;

    const report = `
Diabetes Risk Batch Analysis Report
Generated: ${new Date().toLocaleString()}

=== SUMMARY ===
Total Patients: ${statistics.total}
Average Risk Probability: ${(statistics.avgProbability * 100).toFixed(2)}%

=== RISK DISTRIBUTION ===
Low Risk: ${statistics.lowRisk} (${statistics.lowRiskPct.toFixed(2)}%)
Medium Risk: ${statistics.mediumRisk} (${statistics.mediumRiskPct.toFixed(2)}%)
High Risk: ${statistics.highRisk} (${statistics.highRiskPct.toFixed(2)}%)

=== HIGHEST RISK PATIENTS ===
${statistics.highestRiskPatients
  .map(
    (p, i) =>
      `${i + 1}. ${p.id} - ${((p.prediction?.probability || 0) * 100).toFixed(2)}% (${p.prediction?.risk_level})`
  )
  .join('\n')}

=== RECOMMENDATIONS ===
- ${statistics.highRisk} patients require immediate follow-up
- ${statistics.mediumRisk} patients should be monitored regularly
- Consider lifestyle interventions for patients with modifiable risk factors
`;

    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `batch-summary-${new Date().toISOString()}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Risk distribution chart data
  const riskDistributionData = useMemo(() => {
    if (!statistics) return null;

    return {
      labels: ['Low Risk', 'Medium Risk', 'High Risk'],
      datasets: [
        {
          data: [statistics.lowRisk, statistics.mediumRisk, statistics.highRisk],
          backgroundColor: [
            'rgba(34, 197, 94, 0.8)',
            'rgba(251, 191, 36, 0.8)',
            'rgba(239, 68, 68, 0.8)',
          ],
          borderColor: [
            'rgba(34, 197, 94, 1)',
            'rgba(251, 191, 36, 1)',
            'rgba(239, 68, 68, 1)',
          ],
          borderWidth: 2,
        },
      ],
    };
  }, [statistics]);

  // Probability distribution histogram
  const probabilityHistogramData = useMemo(() => {
    if (processedData.length === 0) return null;

    const bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    const counts = new Array(bins.length - 1).fill(0);

    processedData.forEach((p) => {
      const prob = p.prediction?.probability || 0;
      for (let i = 0; i < bins.length - 1; i++) {
        if (prob >= bins[i] && prob < bins[i + 1]) {
          counts[i]++;
          break;
        }
      }
    });

    return {
      labels: bins.slice(0, -1).map((b, i) => `${(b * 100).toFixed(0)}-${(bins[i + 1] * 100).toFixed(0)}%`),
      datasets: [
        {
          label: 'Number of Patients',
          data: counts,
          backgroundColor: 'rgba(59, 130, 246, 0.6)',
          borderColor: 'rgba(59, 130, 246, 1)',
          borderWidth: 1,
        },
      ],
    };
  }, [processedData]);

  // Render sort icon
  const renderSortIcon = (column: SortColumn) => {
    if (sortColumn !== column) {
      return <FiChevronDown className="w-4 h-4 text-gray-400" />;
    }
    return sortDirection === 'asc' ? (
      <FiChevronUp className="w-4 h-4 text-primary-600" />
    ) : (
      <FiChevronDown className="w-4 h-4 text-primary-600" />
    );
  };

  // Get risk color
  const getRiskColor = (riskLevel: string) => {
    const level = riskLevel.toLowerCase();
    if (level === 'low') return 'text-success-600 bg-success-100 dark:bg-success-900/30';
    if (level === 'medium') return 'text-warning-600 bg-warning-100 dark:bg-warning-900/30';
    if (level === 'high') return 'text-danger-600 bg-danger-100 dark:bg-danger-900/30';
    return 'text-gray-600 bg-gray-100';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Batch Analysis</h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Upload a CSV file with multiple patient records for batch prediction
        </p>
      </div>

      {/* File Upload Section */}
      {!processedData.length && (
        <Card>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Upload CSV File
              </h3>
              {csvFile && (
                <Button variant="secondary" size="sm" onClick={clearAll} icon={<FiX />}>
                  Clear
                </Button>
              )}
            </div>

            {/* Dropzone */}
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors cursor-pointer ${
                isDragActive
                  ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                  : 'border-gray-300 dark:border-gray-700 hover:border-primary-400 dark:hover:border-primary-600'
              }`}
            >
              <input {...getInputProps()} />
              <FiUpload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              {isDragActive ? (
                <p className="text-lg text-gray-600 dark:text-gray-400">Drop the CSV file here...</p>
              ) : (
                <>
                  <p className="text-lg text-gray-600 dark:text-gray-400 mb-2">
                    Drag and drop a CSV file here, or click to select
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-500">
                    CSV must include: {EXPECTED_COLUMNS.join(', ')}
                  </p>
                </>
              )}
            </div>

            {/* File info */}
            {csvFile && !csvError && (
              <div className="flex items-center gap-3 p-4 bg-success-50 dark:bg-success-900/20 border border-success-200 dark:border-success-800 rounded-lg">
                <FiCheckCircle className="w-5 h-5 text-success-600" />
                <div className="flex-1">
                  <p className="font-medium text-gray-900 dark:text-gray-100">{csvFile.name}</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {csvData.length} patients loaded
                  </p>
                </div>
              </div>
            )}

            {/* Error message */}
            {csvError && (
              <div className="flex items-start gap-3 p-4 bg-danger-50 dark:bg-danger-900/20 border border-danger-200 dark:border-danger-800 rounded-lg">
                <FiAlertCircle className="w-5 h-5 text-danger-600 mt-0.5" />
                <div className="flex-1">
                  <p className="font-medium text-danger-900 dark:text-danger-100">Error</p>
                  <p className="text-sm text-danger-700 dark:text-danger-300">{csvError}</p>
                </div>
              </div>
            )}

            {/* CSV Format Help */}
            <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">
                CSV Format Requirements
              </h4>
              <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1 list-disc list-inside">
                <li>First row must contain column headers (case-sensitive)</li>
                <li>Required columns: {EXPECTED_COLUMNS.join(', ')}</li>
                <li>All values must be numeric</li>
                <li>No missing values allowed</li>
              </ul>
              <div className="mt-3">
                <p className="text-xs text-gray-500 dark:text-gray-500 mb-2">Example CSV format:</p>
                <pre className="text-xs bg-white dark:bg-gray-900 p-2 rounded border border-gray-200 dark:border-gray-700 overflow-x-auto">
                  {`Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
6,148,72,35,0,33.6,0.627,50
1,85,66,29,0,26.6,0.351,31`}
                </pre>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* Preview Section */}
      {showPreview && csvData.length > 0 && (
        <Card>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  Preview - First 5 Rows
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Review the data before processing
                </p>
              </div>
              <Button
                variant="primary"
                onClick={processBatch}
                disabled={processing}
                loading={processing}
                icon={<FiBarChart2 />}
              >
                {processing ? `Processing... ${processProgress.toFixed(0)}%` : 'Process Batch'}
              </Button>
            </div>

            {/* Preview table */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 dark:bg-gray-800">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium text-gray-700 dark:text-gray-300">
                      ID
                    </th>
                    {EXPECTED_COLUMNS.map((col) => (
                      <th
                        key={col}
                        className="px-3 py-2 text-left font-medium text-gray-700 dark:text-gray-300"
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {csvData.slice(0, 5).map((patient) => (
                    <tr
                      key={patient.id}
                      className="border-t border-gray-200 dark:border-gray-700"
                    >
                      <td className="px-3 py-2 text-gray-900 dark:text-gray-100">{patient.id}</td>
                      <td className="px-3 py-2 text-gray-600 dark:text-gray-400">
                        {patient.Pregnancies}
                      </td>
                      <td className="px-3 py-2 text-gray-600 dark:text-gray-400">
                        {patient.Glucose}
                      </td>
                      <td className="px-3 py-2 text-gray-600 dark:text-gray-400">
                        {patient.BloodPressure}
                      </td>
                      <td className="px-3 py-2 text-gray-600 dark:text-gray-400">
                        {patient.SkinThickness}
                      </td>
                      <td className="px-3 py-2 text-gray-600 dark:text-gray-400">
                        {patient.Insulin}
                      </td>
                      <td className="px-3 py-2 text-gray-600 dark:text-gray-400">
                        {patient.BMI}
                      </td>
                      <td className="px-3 py-2 text-gray-600 dark:text-gray-400">
                        {patient.DiabetesPedigreeFunction}
                      </td>
                      <td className="px-3 py-2 text-gray-600 dark:text-gray-400">{patient.Age}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {csvData.length > 5 && (
              <p className="text-sm text-gray-500 dark:text-gray-500 text-center">
                ... and {csvData.length - 5} more rows
              </p>
            )}
          </div>
        </Card>
      )}

      {/* Processing Progress */}
      {processing && (
        <Card>
          <div className="text-center py-8">
            <LoadingSpinner size="lg" />
            <p className="text-lg font-medium text-gray-900 dark:text-gray-100 mt-4">
              Processing Batch Predictions
            </p>
            <p className="text-gray-600 dark:text-gray-400 mt-2">
              {processProgress.toFixed(0)}% complete
            </p>
            <div className="w-full max-w-md mx-auto mt-4 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-primary-600 transition-all duration-300"
                style={{ width: `${processProgress}%` }}
              />
            </div>
          </div>
        </Card>
      )}

      {/* Results Section */}
      {processedData.length > 0 && statistics && (
        <>
          {/* Statistics Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20">
              <div>
                <p className="text-sm font-medium text-blue-600 dark:text-blue-400">
                  Total Patients
                </p>
                <p className="text-3xl font-bold text-blue-900 dark:text-blue-100 mt-1">
                  {statistics.total}
                </p>
              </div>
            </Card>

            <Card className="bg-gradient-to-br from-success-50 to-success-100 dark:from-success-900/20 dark:to-success-800/20">
              <div>
                <p className="text-sm font-medium text-success-600 dark:text-success-400">
                  Low Risk
                </p>
                <p className="text-3xl font-bold text-success-900 dark:text-success-100 mt-1">
                  {statistics.lowRisk}
                  <span className="text-base ml-2">({statistics.lowRiskPct.toFixed(1)}%)</span>
                </p>
              </div>
            </Card>

            <Card className="bg-gradient-to-br from-warning-50 to-warning-100 dark:from-warning-900/20 dark:to-warning-800/20">
              <div>
                <p className="text-sm font-medium text-warning-600 dark:text-warning-400">
                  Medium Risk
                </p>
                <p className="text-3xl font-bold text-warning-900 dark:text-warning-100 mt-1">
                  {statistics.mediumRisk}
                  <span className="text-base ml-2">({statistics.mediumRiskPct.toFixed(1)}%)</span>
                </p>
              </div>
            </Card>

            <Card className="bg-gradient-to-br from-danger-50 to-danger-100 dark:from-danger-900/20 dark:to-danger-800/20">
              <div>
                <p className="text-sm font-medium text-danger-600 dark:text-danger-400">
                  High Risk
                </p>
                <p className="text-3xl font-bold text-danger-900 dark:text-danger-100 mt-1">
                  {statistics.highRisk}
                  <span className="text-base ml-2">({statistics.highRiskPct.toFixed(1)}%)</span>
                </p>
              </div>
            </Card>
          </div>

          {/* Visualizations */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Risk Distribution Pie Chart */}
            {riskDistributionData && (
              <Card title="Risk Distribution">
                <div className="h-64">
                  <Pie
                    data={riskDistributionData}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          position: 'bottom',
                        },
                      },
                    }}
                  />
                </div>
              </Card>
            )}

            {/* Probability Histogram */}
            {probabilityHistogramData && (
              <Card title="Risk Probability Distribution">
                <div className="h-64">
                  <Bar
                    data={probabilityHistogramData}
                    options={{
                      responsive: true,
                      maintainAspectRatio: false,
                      plugins: {
                        legend: {
                          display: false,
                        },
                      },
                      scales: {
                        y: {
                          beginAtZero: true,
                          ticks: {
                            stepSize: 1,
                          },
                        },
                      },
                    }}
                  />
                </div>
              </Card>
            )}
          </div>

          {/* Highest Risk Patients */}
          <Card title="Highest Risk Patients" subtitle="Top 5 patients requiring immediate attention">
            <div className="space-y-2">
              {statistics.highestRiskPatients.map((patient, index) => (
                <div
                  key={patient.id}
                  className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-danger-100 dark:bg-danger-900/30 flex items-center justify-center">
                      <span className="text-sm font-bold text-danger-600">{index + 1}</span>
                    </div>
                    <div>
                      <p className="font-medium text-gray-900 dark:text-gray-100">{patient.id}</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        Age: {patient.Age}, BMI: {patient.BMI}, Glucose: {patient.Glucose}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span
                      className={`px-3 py-1 rounded-full text-sm font-medium ${getRiskColor(
                        patient.prediction?.risk_level || ''
                      )}`}
                    >
                      {((patient.prediction?.probability || 0) * 100).toFixed(1)}%
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => viewPatientDetails(patient)}
                      icon={<FiEye />}
                    >
                      View
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Results Table */}
          <Card>
            <div className="space-y-4">
              {/* Table controls */}
              <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
                <div className="flex-1 w-full sm:w-auto">
                  <div className="relative">
                    <FiSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Search by ID or risk level..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="input pl-10 w-full"
                    />
                  </div>
                </div>

                <div className="flex gap-2 flex-wrap">
                  <div className="flex items-center gap-2">
                    <FiFilter className="text-gray-400" />
                    <select
                      value={riskFilter}
                      onChange={(e) => setRiskFilter(e.target.value)}
                      className="input"
                    >
                      <option value="all">All Risks</option>
                      <option value="low">Low Risk</option>
                      <option value="medium">Medium Risk</option>
                      <option value="high">High Risk</option>
                    </select>
                  </div>

                  <Button variant="outline" size="sm" onClick={exportResultsCSV} icon={<FiDownload />}>
                    Export CSV
                  </Button>

                  <Button
                    variant="outline"
                    size="sm"
                    onClick={exportSummaryReport}
                    icon={<FiFileText />}
                  >
                    Export Report
                  </Button>

                  <Button variant="secondary" size="sm" onClick={clearAll} icon={<FiX />}>
                    New Batch
                  </Button>
                </div>
              </div>

              {/* Results count */}
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Showing {filteredAndSortedData.length} of {processedData.length} patients
              </p>

              {/* Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 dark:bg-gray-800">
                    <tr>
                      <th className="px-3 py-3 text-left font-medium text-gray-700 dark:text-gray-300">
                        ID
                      </th>
                      <th className="px-3 py-3 text-left font-medium text-gray-700 dark:text-gray-300">
                        <button
                          onClick={() => handleSort('Age')}
                          className="flex items-center gap-1 hover:text-primary-600"
                        >
                          Age {renderSortIcon('Age')}
                        </button>
                      </th>
                      <th className="px-3 py-3 text-left font-medium text-gray-700 dark:text-gray-300">
                        <button
                          onClick={() => handleSort('Glucose')}
                          className="flex items-center gap-1 hover:text-primary-600"
                        >
                          Glucose {renderSortIcon('Glucose')}
                        </button>
                      </th>
                      <th className="px-3 py-3 text-left font-medium text-gray-700 dark:text-gray-300">
                        <button
                          onClick={() => handleSort('BMI')}
                          className="flex items-center gap-1 hover:text-primary-600"
                        >
                          BMI {renderSortIcon('BMI')}
                        </button>
                      </th>
                      <th className="px-3 py-3 text-left font-medium text-gray-700 dark:text-gray-300">
                        <button
                          onClick={() => handleSort('prediction')}
                          className="flex items-center gap-1 hover:text-primary-600"
                        >
                          Prediction {renderSortIcon('prediction')}
                        </button>
                      </th>
                      <th className="px-3 py-3 text-left font-medium text-gray-700 dark:text-gray-300">
                        <button
                          onClick={() => handleSort('probability')}
                          className="flex items-center gap-1 hover:text-primary-600"
                        >
                          Probability {renderSortIcon('probability')}
                        </button>
                      </th>
                      <th className="px-3 py-3 text-left font-medium text-gray-700 dark:text-gray-300">
                        <button
                          onClick={() => handleSort('risk_level')}
                          className="flex items-center gap-1 hover:text-primary-600"
                        >
                          Risk Level {renderSortIcon('risk_level')}
                        </button>
                      </th>
                      <th className="px-3 py-3 text-left font-medium text-gray-700 dark:text-gray-300">
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredAndSortedData.map((patient) => (
                      <tr
                        key={patient.id}
                        className="border-t border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                      >
                        <td className="px-3 py-3 font-medium text-gray-900 dark:text-gray-100">
                          {patient.id}
                        </td>
                        <td className="px-3 py-3 text-gray-600 dark:text-gray-400">
                          {patient.Age}
                        </td>
                        <td className="px-3 py-3 text-gray-600 dark:text-gray-400">
                          {patient.Glucose}
                        </td>
                        <td className="px-3 py-3 text-gray-600 dark:text-gray-400">
                          {patient.BMI}
                        </td>
                        <td className="px-3 py-3 text-gray-600 dark:text-gray-400">
                          {patient.prediction?.prediction_label}
                        </td>
                        <td className="px-3 py-3 text-gray-600 dark:text-gray-400">
                          {((patient.prediction?.probability || 0) * 100).toFixed(2)}%
                        </td>
                        <td className="px-3 py-3">
                          <span
                            className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskColor(
                              patient.prediction?.risk_level || ''
                            )}`}
                          >
                            {patient.prediction?.risk_level}
                          </span>
                        </td>
                        <td className="px-3 py-3">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => viewPatientDetails(patient)}
                            icon={<FiEye />}
                          >
                            Details
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {filteredAndSortedData.length === 0 && (
                <div className="text-center py-12">
                  <p className="text-gray-500 dark:text-gray-400">
                    No patients match your search criteria
                  </p>
                </div>
              )}
            </div>
          </Card>
        </>
      )}

      {/* Patient Detail Modal */}
      {selectedPatient && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-gray-900 rounded-lg shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-y-auto">
            <div className="sticky top-0 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 px-6 py-4 flex items-center justify-between z-10">
              <div>
                <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                  Patient Details - {selectedPatient.id}
                </h2>
                <p className="text-gray-600 dark:text-gray-400 mt-1">
                  Comprehensive analysis and recommendations
                </p>
              </div>
              <button
                onClick={() => setSelectedPatient(null)}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
              >
                <FiX className="w-6 h-6 text-gray-600 dark:text-gray-400" />
              </button>
            </div>

            <div className="p-6">
              {loadingDetails ? (
                <div className="text-center py-12">
                  <LoadingSpinner size="lg" />
                  <p className="text-gray-600 dark:text-gray-400 mt-4">
                    Loading detailed analysis...
                  </p>
                </div>
              ) : selectedPatient.detailedPrediction ? (
                <PredictionResults result={selectedPatient.detailedPrediction} />
              ) : (
                <div className="text-center py-12">
                  <p className="text-gray-600 dark:text-gray-400">
                    Detailed analysis not available
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BatchAnalysis;
