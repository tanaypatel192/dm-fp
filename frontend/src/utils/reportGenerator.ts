/**
 * PDF Report Generator Utility
 *
 * Generates comprehensive medical reports using jsPDF and jsPDF-autotable
 * Supports single patient, batch, and model comparison reports
 */

import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import html2canvas from 'html2canvas';
import type {
  PatientInput,
  ComprehensivePredictionOutput,
  ModelMetrics,
  PredictionOutput,
} from '@/types/api';

// Extend jsPDF type to include autoTable
declare module 'jspdf' {
  interface jsPDF {
    autoTable: typeof autoTable;
    lastAutoTable?: {
      finalY: number;
    };
  }
}

// Report configuration
const REPORT_CONFIG = {
  margin: {
    top: 20,
    right: 15,
    bottom: 20,
    left: 15,
  },
  colors: {
    primary: [41, 98, 255] as [number, number, number],
    success: [34, 197, 94] as [number, number, number],
    warning: [251, 191, 36] as [number, number, number],
    danger: [239, 68, 68] as [number, number, number],
    gray: [107, 114, 128] as [number, number, number],
    lightGray: [243, 244, 246] as [number, number, number],
  },
  fonts: {
    title: 18,
    heading: 14,
    subheading: 12,
    body: 10,
    small: 8,
  },
};

/**
 * Convert HTML element to canvas for embedding in PDF
 */
export async function elementToCanvas(element: HTMLElement): Promise<HTMLCanvasElement> {
  return html2canvas(element, {
    scale: 2,
    logging: false,
    useCORS: true,
  });
}

/**
 * Add header to PDF page
 */
function addHeader(doc: jsPDF, title: string): number {
  const pageWidth = doc.internal.pageSize.getWidth();

  // Logo placeholder (you can add actual logo here)
  doc.setFillColor(...REPORT_CONFIG.colors.primary);
  doc.rect(REPORT_CONFIG.margin.left, REPORT_CONFIG.margin.top - 5, 30, 10, 'F');
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(10);
  doc.text('DM-FP', REPORT_CONFIG.margin.left + 15, REPORT_CONFIG.margin.top, { align: 'center' });

  // Title
  doc.setTextColor(0, 0, 0);
  doc.setFontSize(REPORT_CONFIG.fonts.title);
  doc.setFont('helvetica', 'bold');
  doc.text(title, pageWidth / 2, REPORT_CONFIG.margin.top, { align: 'center' });

  // Date
  doc.setFontSize(REPORT_CONFIG.fonts.small);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(...REPORT_CONFIG.colors.gray);
  const dateStr = new Date().toLocaleString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
  doc.text(`Generated: ${dateStr}`, pageWidth - REPORT_CONFIG.margin.right, REPORT_CONFIG.margin.top, {
    align: 'right',
  });

  // Horizontal line
  doc.setDrawColor(...REPORT_CONFIG.colors.lightGray);
  doc.setLineWidth(0.5);
  doc.line(
    REPORT_CONFIG.margin.left,
    REPORT_CONFIG.margin.top + 5,
    pageWidth - REPORT_CONFIG.margin.right,
    REPORT_CONFIG.margin.top + 5
  );

  return REPORT_CONFIG.margin.top + 15;
}

/**
 * Add footer to PDF page
 */
function addFooter(doc: jsPDF, pageNumber: number, totalPages: number) {
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const footerY = pageHeight - REPORT_CONFIG.margin.bottom + 10;

  // Horizontal line
  doc.setDrawColor(...REPORT_CONFIG.colors.lightGray);
  doc.setLineWidth(0.5);
  doc.line(
    REPORT_CONFIG.margin.left,
    footerY - 5,
    pageWidth - REPORT_CONFIG.margin.right,
    footerY - 5
  );

  // Footer text
  doc.setFontSize(REPORT_CONFIG.fonts.small);
  doc.setFont('helvetica', 'normal');
  doc.setTextColor(...REPORT_CONFIG.colors.gray);
  doc.text(
    'Diabetes Prediction System - For Medical Professional Use Only',
    REPORT_CONFIG.margin.left,
    footerY
  );

  // Page number
  doc.text(`Page ${pageNumber} of ${totalPages}`, pageWidth - REPORT_CONFIG.margin.right, footerY, {
    align: 'right',
  });
}

/**
 * Add section heading
 */
function addSectionHeading(doc: jsPDF, text: string, y: number): number {
  doc.setFontSize(REPORT_CONFIG.fonts.heading);
  doc.setFont('helvetica', 'bold');
  doc.setTextColor(0, 0, 0);
  doc.text(text, REPORT_CONFIG.margin.left, y);

  // Underline
  const textWidth = doc.getTextWidth(text);
  doc.setDrawColor(...REPORT_CONFIG.colors.primary);
  doc.setLineWidth(0.3);
  doc.line(REPORT_CONFIG.margin.left, y + 1, REPORT_CONFIG.margin.left + textWidth, y + 1);

  return y + 8;
}

/**
 * Add text content
 */
function addText(doc: jsPDF, text: string, y: number, options?: { bold?: boolean; color?: [number, number, number] }): number {
  doc.setFontSize(REPORT_CONFIG.fonts.body);
  doc.setFont('helvetica', options?.bold ? 'bold' : 'normal');
  doc.setTextColor(...(options?.color || [0, 0, 0]));

  const pageWidth = doc.internal.pageSize.getWidth();
  const maxWidth = pageWidth - REPORT_CONFIG.margin.left - REPORT_CONFIG.margin.right;
  const lines = doc.splitTextToSize(text, maxWidth);

  doc.text(lines, REPORT_CONFIG.margin.left, y);

  return y + (lines.length * 5);
}

/**
 * Get risk color based on risk level
 */
function getRiskColor(riskLevel: string): [number, number, number] {
  const level = riskLevel.toLowerCase();
  if (level === 'low') return REPORT_CONFIG.colors.success;
  if (level === 'medium') return REPORT_CONFIG.colors.warning;
  if (level === 'high') return REPORT_CONFIG.colors.danger;
  return REPORT_CONFIG.colors.gray;
}

/**
 * Format patient data for display
 */
function formatPatientData(data: Record<string, any>): Array<[string, string]> {
  return [
    ['Pregnancies', String(data.Pregnancies || 0)],
    ['Glucose (mg/dL)', String(data.Glucose || 0)],
    ['Blood Pressure (mm Hg)', String(data.BloodPressure || 0)],
    ['Skin Thickness (mm)', String(data.SkinThickness || 0)],
    ['Insulin (μU/mL)', String(data.Insulin || 0)],
    ['BMI (kg/m²)', String(data.BMI?.toFixed(1) || 0)],
    ['Diabetes Pedigree Function', String(data.DiabetesPedigreeFunction?.toFixed(3) || 0)],
    ['Age (years)', String(data.Age || 0)],
  ];
}

/**
 * Generate Single Patient Report
 */
export async function generateSinglePatientReport(
  patientData: PatientInput,
  prediction: ComprehensivePredictionOutput,
  options?: {
    includeCharts?: boolean;
    chartElements?: {
      shapChart?: HTMLElement;
      riskChart?: HTMLElement;
    };
  }
): Promise<jsPDF> {
  const doc = new jsPDF();
  let currentY = addHeader(doc, 'Diabetes Risk Assessment Report');

  // Patient Information Section
  currentY = addSectionHeading(doc, '1. Patient Information', currentY + 5);

  doc.autoTable({
    startY: currentY,
    head: [['Parameter', 'Value']],
    body: formatPatientData(patientData),
    theme: 'striped',
    headStyles: { fillColor: REPORT_CONFIG.colors.primary },
    margin: { left: REPORT_CONFIG.margin.left, right: REPORT_CONFIG.margin.right },
  });

  currentY = doc.lastAutoTable?.finalY || currentY + 50;

  // Risk Assessment Section
  currentY = addSectionHeading(doc, '2. Risk Assessment', currentY + 10);

  const riskColor = getRiskColor(prediction.risk_level);

  // Risk summary box
  doc.setFillColor(...riskColor);
  doc.setDrawColor(...riskColor);
  doc.roundedRect(REPORT_CONFIG.margin.left, currentY, 180, 25, 3, 3, 'FD');

  doc.setTextColor(255, 255, 255);
  doc.setFontSize(REPORT_CONFIG.fonts.heading);
  doc.setFont('helvetica', 'bold');
  doc.text(`Risk Level: ${prediction.risk_level.toUpperCase()}`, REPORT_CONFIG.margin.left + 5, currentY + 8);

  doc.setFontSize(REPORT_CONFIG.fonts.body);
  doc.text(
    `Probability: ${(prediction.ensemble_probability * 100).toFixed(1)}%`,
    REPORT_CONFIG.margin.left + 5,
    currentY + 15
  );
  doc.text(
    `Confidence: ${(prediction.ensemble_confidence * 100).toFixed(1)}%`,
    REPORT_CONFIG.margin.left + 5,
    currentY + 21
  );

  currentY += 35;

  // Model Predictions Section
  currentY = addSectionHeading(doc, '3. Model Predictions', currentY);

  const modelTableData = prediction.model_predictions.map((pred) => [
    pred.model_name,
    pred.prediction_label,
    `${(pred.probability * 100).toFixed(1)}%`,
    `${(pred.confidence * 100).toFixed(1)}%`,
  ]);

  doc.autoTable({
    startY: currentY,
    head: [['Model', 'Prediction', 'Probability', 'Confidence']],
    body: modelTableData,
    theme: 'grid',
    headStyles: { fillColor: REPORT_CONFIG.colors.primary },
    margin: { left: REPORT_CONFIG.margin.left, right: REPORT_CONFIG.margin.right },
  });

  currentY = doc.lastAutoTable?.finalY || currentY + 40;

  // Add new page for detailed analysis
  doc.addPage();
  currentY = addHeader(doc, 'Diabetes Risk Assessment Report (Continued)');

  // Risk Factors Section
  currentY = addSectionHeading(doc, '4. Risk Factors', currentY + 5);

  if (prediction.risk_factors && prediction.risk_factors.length > 0) {
    const riskFactorsData = prediction.risk_factors.map((factor) => [
      factor.factor,
      String(factor.current_value),
      factor.risk_level,
      factor.is_modifiable ? 'Yes' : 'No',
    ]);

    doc.autoTable({
      startY: currentY,
      head: [['Factor', 'Current Value', 'Risk Level', 'Modifiable']],
      body: riskFactorsData,
      theme: 'striped',
      headStyles: { fillColor: REPORT_CONFIG.colors.primary },
      margin: { left: REPORT_CONFIG.margin.left, right: REPORT_CONFIG.margin.right },
      columnStyles: {
        2: {
          cellWidth: 30,
          halign: 'center',
        },
        3: {
          cellWidth: 25,
          halign: 'center',
        },
      },
    });

    currentY = doc.lastAutoTable?.finalY || currentY + 40;
  }

  // Recommendations Section
  currentY = addSectionHeading(doc, '5. Personalized Recommendations', currentY + 10);

  if (prediction.recommendations && prediction.recommendations.length > 0) {
    prediction.recommendations.forEach((rec, index) => {
      const priorityColor = rec.priority === 'High' ? REPORT_CONFIG.colors.danger :
                            rec.priority === 'Medium' ? REPORT_CONFIG.colors.warning :
                            REPORT_CONFIG.colors.success;

      // Priority badge
      doc.setFillColor(...priorityColor);
      doc.roundedRect(REPORT_CONFIG.margin.left, currentY - 3, 20, 5, 1, 1, 'F');
      doc.setTextColor(255, 255, 255);
      doc.setFontSize(REPORT_CONFIG.fonts.small);
      doc.text(rec.priority, REPORT_CONFIG.margin.left + 10, currentY, { align: 'center' });

      // Category and recommendation
      doc.setTextColor(0, 0, 0);
      doc.setFontSize(REPORT_CONFIG.fonts.body);
      doc.setFont('helvetica', 'bold');
      doc.text(`${index + 1}. ${rec.category}`, REPORT_CONFIG.margin.left + 25, currentY);

      currentY += 5;
      doc.setFont('helvetica', 'normal');
      currentY = addText(doc, rec.recommendation, currentY);

      doc.setTextColor(...REPORT_CONFIG.colors.gray);
      doc.setFontSize(REPORT_CONFIG.fonts.small);
      currentY = addText(doc, `Rationale: ${rec.rationale}`, currentY + 1);

      currentY += 5;

      // Check if we need a new page
      if (currentY > doc.internal.pageSize.getHeight() - 40) {
        doc.addPage();
        currentY = addHeader(doc, 'Diabetes Risk Assessment Report (Continued)');
        currentY += 10;
      }
    });
  }

  // Add charts if provided
  if (options?.includeCharts && options.chartElements) {
    doc.addPage();
    currentY = addHeader(doc, 'Diabetes Risk Assessment Report (Continued)');

    currentY = addSectionHeading(doc, '6. Visual Analysis', currentY + 5);

    if (options.chartElements.shapChart) {
      try {
        const canvas = await elementToCanvas(options.chartElements.shapChart);
        const imgData = canvas.toDataURL('image/png');
        doc.addImage(imgData, 'PNG', REPORT_CONFIG.margin.left, currentY, 180, 80);
        currentY += 85;
      } catch (error) {
        console.error('Error adding SHAP chart:', error);
      }
    }

    if (options.chartElements.riskChart && currentY < doc.internal.pageSize.getHeight() - 100) {
      try {
        const canvas = await elementToCanvas(options.chartElements.riskChart);
        const imgData = canvas.toDataURL('image/png');
        doc.addImage(imgData, 'PNG', REPORT_CONFIG.margin.left, currentY, 180, 80);
        currentY += 85;
      } catch (error) {
        console.error('Error adding risk chart:', error);
      }
    }
  }

  // Disclaimer
  doc.addPage();
  currentY = addHeader(doc, 'Diabetes Risk Assessment Report (Continued)');

  currentY = addSectionHeading(doc, 'Disclaimer', currentY + 5);

  doc.setFontSize(REPORT_CONFIG.fonts.small);
  doc.setTextColor(...REPORT_CONFIG.colors.gray);

  const disclaimer = `This report is generated by an AI-powered diabetes risk prediction system and is intended for use by qualified healthcare professionals only. The predictions and recommendations provided should not be used as the sole basis for medical decisions. Always consult with a qualified healthcare provider for proper diagnosis and treatment. The system uses machine learning models trained on historical data and may not account for all individual patient circumstances. Results should be interpreted in the context of a comprehensive clinical evaluation.`;

  currentY = addText(doc, disclaimer, currentY);

  currentY += 10;
  currentY = addSectionHeading(doc, 'Contact Information', currentY);

  currentY = addText(doc, 'For questions or concerns about this report:', currentY);
  currentY = addText(doc, 'Email: support@diabetes-prediction.example.com', currentY + 2);
  currentY = addText(doc, 'Phone: +1 (555) 123-4567', currentY + 4);

  // Add footers to all pages
  const totalPages = doc.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    addFooter(doc, i, totalPages);
  }

  return doc;
}

/**
 * Generate Batch Report
 */
export async function generateBatchReport(
  batchResults: Array<{
    id: string;
    patientData: PatientInput;
    prediction: PredictionOutput;
  }>,
  summary: {
    totalPatients: number;
    lowRisk: number;
    mediumRisk: number;
    highRisk: number;
    avgProbability: number;
  }
): Promise<jsPDF> {
  const doc = new jsPDF();
  let currentY = addHeader(doc, 'Batch Diabetes Risk Assessment Report');

  // Executive Summary
  currentY = addSectionHeading(doc, 'Executive Summary', currentY + 5);

  const summaryData = [
    ['Total Patients Analyzed', String(summary.totalPatients)],
    ['Low Risk Patients', `${summary.lowRisk} (${((summary.lowRisk / summary.totalPatients) * 100).toFixed(1)}%)`],
    ['Medium Risk Patients', `${summary.mediumRisk} (${((summary.mediumRisk / summary.totalPatients) * 100).toFixed(1)}%)`],
    ['High Risk Patients', `${summary.highRisk} (${((summary.highRisk / summary.totalPatients) * 100).toFixed(1)}%)`],
    ['Average Risk Probability', `${(summary.avgProbability * 100).toFixed(1)}%`],
  ];

  doc.autoTable({
    startY: currentY,
    body: summaryData,
    theme: 'grid',
    columnStyles: {
      0: { fontStyle: 'bold', cellWidth: 80 },
      1: { halign: 'right', cellWidth: 100 },
    },
    margin: { left: REPORT_CONFIG.margin.left, right: REPORT_CONFIG.margin.right },
  });

  currentY = doc.lastAutoTable?.finalY || currentY + 50;

  // Risk Distribution Visualization (text-based)
  currentY = addSectionHeading(doc, 'Risk Distribution', currentY + 10);

  const maxBarWidth = 150;
  const lowWidth = (summary.lowRisk / summary.totalPatients) * maxBarWidth;
  const mediumWidth = (summary.mediumRisk / summary.totalPatients) * maxBarWidth;
  const highWidth = (summary.highRisk / summary.totalPatients) * maxBarWidth;

  // Low Risk
  doc.setFillColor(...REPORT_CONFIG.colors.success);
  doc.rect(REPORT_CONFIG.margin.left, currentY, lowWidth, 8, 'F');
  doc.setTextColor(0, 0, 0);
  doc.setFontSize(REPORT_CONFIG.fonts.small);
  doc.text(`Low: ${summary.lowRisk}`, REPORT_CONFIG.margin.left + lowWidth + 2, currentY + 5);

  currentY += 12;

  // Medium Risk
  doc.setFillColor(...REPORT_CONFIG.colors.warning);
  doc.rect(REPORT_CONFIG.margin.left, currentY, mediumWidth, 8, 'F');
  doc.text(`Medium: ${summary.mediumRisk}`, REPORT_CONFIG.margin.left + mediumWidth + 2, currentY + 5);

  currentY += 12;

  // High Risk
  doc.setFillColor(...REPORT_CONFIG.colors.danger);
  doc.rect(REPORT_CONFIG.margin.left, currentY, highWidth, 8, 'F');
  doc.text(`High: ${summary.highRisk}`, REPORT_CONFIG.margin.left + highWidth + 2, currentY + 5);

  currentY += 20;

  // Detailed Results Table
  doc.addPage();
  currentY = addHeader(doc, 'Batch Diabetes Risk Assessment Report (Continued)');

  currentY = addSectionHeading(doc, 'Detailed Results', currentY + 5);

  const resultsData = batchResults.map((result) => [
    result.id,
    String(result.patientData.Age),
    String(result.patientData.BMI.toFixed(1)),
    String(result.patientData.Glucose),
    result.prediction.prediction_label,
    `${(result.prediction.probability * 100).toFixed(1)}%`,
    result.prediction.risk_level,
  ]);

  doc.autoTable({
    startY: currentY,
    head: [['Patient ID', 'Age', 'BMI', 'Glucose', 'Prediction', 'Probability', 'Risk Level']],
    body: resultsData,
    theme: 'striped',
    headStyles: { fillColor: REPORT_CONFIG.colors.primary },
    margin: { left: REPORT_CONFIG.margin.left, right: REPORT_CONFIG.margin.right },
    styles: { fontSize: 8 },
    columnStyles: {
      0: { cellWidth: 25 },
      1: { cellWidth: 15, halign: 'center' },
      2: { cellWidth: 15, halign: 'center' },
      3: { cellWidth: 20, halign: 'center' },
      4: { cellWidth: 30 },
      5: { cellWidth: 25, halign: 'right' },
      6: { cellWidth: 25, halign: 'center' },
    },
    didParseCell: (data) => {
      // Color code risk levels
      if (data.column.index === 6 && data.section === 'body') {
        const riskLevel = data.cell.text[0].toLowerCase();
        if (riskLevel === 'low') {
          data.cell.styles.fillColor = [34, 197, 94, 0.2] as any;
          data.cell.styles.textColor = REPORT_CONFIG.colors.success;
        } else if (riskLevel === 'medium') {
          data.cell.styles.fillColor = [251, 191, 36, 0.2] as any;
          data.cell.styles.textColor = REPORT_CONFIG.colors.warning;
        } else if (riskLevel === 'high') {
          data.cell.styles.fillColor = [239, 68, 68, 0.2] as any;
          data.cell.styles.textColor = REPORT_CONFIG.colors.danger;
        }
      }
    },
  });

  // Recommendations
  doc.addPage();
  currentY = addHeader(doc, 'Batch Diabetes Risk Assessment Report (Continued)');

  currentY = addSectionHeading(doc, 'Batch Recommendations', currentY + 5);

  currentY = addText(
    doc,
    `Based on the analysis of ${summary.totalPatients} patients, the following actions are recommended:`,
    currentY
  );

  currentY += 8;

  if (summary.highRisk > 0) {
    currentY = addText(
      doc,
      `• ${summary.highRisk} patients identified as HIGH RISK require immediate follow-up and intervention.`,
      currentY,
      { color: REPORT_CONFIG.colors.danger }
    );
    currentY += 5;
  }

  if (summary.mediumRisk > 0) {
    currentY = addText(
      doc,
      `• ${summary.mediumRisk} patients at MEDIUM RISK should be monitored regularly and counseled on lifestyle modifications.`,
      currentY,
      { color: REPORT_CONFIG.colors.warning }
    );
    currentY += 5;
  }

  currentY = addText(
    doc,
    '• Consider implementing a comprehensive diabetes prevention program for at-risk populations.',
    currentY
  );
  currentY += 5;

  currentY = addText(
    doc,
    '• Schedule follow-up screenings based on individual risk profiles.',
    currentY
  );
  currentY += 5;

  currentY = addText(
    doc,
    '• Provide educational resources on diabetes prevention and management to all patients.',
    currentY
  );

  // Add footers
  const totalPages = doc.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    addFooter(doc, i, totalPages);
  }

  return doc;
}

/**
 * Generate Model Comparison Report
 */
export async function generateModelComparisonReport(
  models: ModelMetrics[],
  comparisonData?: {
    testCases: Array<{
      case: string;
      predictions: Record<string, string>;
      agreement: boolean;
    }>;
  }
): Promise<jsPDF> {
  const doc = new jsPDF();
  let currentY = addHeader(doc, 'Model Performance Comparison Report');

  // Model Metrics Section
  currentY = addSectionHeading(doc, '1. Model Performance Metrics', currentY + 5);

  const metricsData = models.map((model) => [
    model.model_name,
    `${(model.accuracy * 100).toFixed(2)}%`,
    `${(model.precision * 100).toFixed(2)}%`,
    `${(model.recall * 100).toFixed(2)}%`,
    `${(model.f1_score * 100).toFixed(2)}%`,
    `${(model.roc_auc * 100).toFixed(2)}%`,
  ]);

  doc.autoTable({
    startY: currentY,
    head: [['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']],
    body: metricsData,
    theme: 'grid',
    headStyles: { fillColor: REPORT_CONFIG.colors.primary },
    margin: { left: REPORT_CONFIG.margin.left, right: REPORT_CONFIG.margin.right },
    columnStyles: {
      1: { halign: 'right' },
      2: { halign: 'right' },
      3: { halign: 'right' },
      4: { halign: 'right' },
      5: { halign: 'right' },
    },
  });

  currentY = doc.lastAutoTable?.finalY || currentY + 50;

  // Model Status
  currentY = addSectionHeading(doc, '2. Model Status', currentY + 10);

  const statusData = models.map((model) => [
    model.model_name,
    model.is_loaded ? 'Loaded' : 'Not Loaded',
    model.last_updated || 'N/A',
  ]);

  doc.autoTable({
    startY: currentY,
    head: [['Model', 'Status', 'Last Updated']],
    body: statusData,
    theme: 'striped',
    headStyles: { fillColor: REPORT_CONFIG.colors.primary },
    margin: { left: REPORT_CONFIG.margin.left, right: REPORT_CONFIG.margin.right },
    didParseCell: (data) => {
      if (data.column.index === 1 && data.section === 'body') {
        if (data.cell.text[0] === 'Loaded') {
          data.cell.styles.textColor = REPORT_CONFIG.colors.success;
          data.cell.styles.fontStyle = 'bold';
        } else {
          data.cell.styles.textColor = REPORT_CONFIG.colors.danger;
        }
      }
    },
  });

  currentY = doc.lastAutoTable?.finalY || currentY + 40;

  // Comparison Analysis
  if (comparisonData && comparisonData.testCases.length > 0) {
    doc.addPage();
    currentY = addHeader(doc, 'Model Performance Comparison Report (Continued)');

    currentY = addSectionHeading(doc, '3. Model Agreement Analysis', currentY + 5);

    const comparisonTableData = comparisonData.testCases.map((test) => {
      const predictions = Object.entries(test.predictions)
        .map(([model, pred]) => `${model}: ${pred}`)
        .join('\n');

      return [
        test.case,
        predictions,
        test.agreement ? 'Yes' : 'No',
      ];
    });

    doc.autoTable({
      startY: currentY,
      head: [['Test Case', 'Model Predictions', 'Agreement']],
      body: comparisonTableData,
      theme: 'grid',
      headStyles: { fillColor: REPORT_CONFIG.colors.primary },
      margin: { left: REPORT_CONFIG.margin.left, right: REPORT_CONFIG.margin.right },
      styles: { fontSize: 9 },
      columnStyles: {
        0: { cellWidth: 40 },
        1: { cellWidth: 100 },
        2: { cellWidth: 25, halign: 'center' },
      },
      didParseCell: (data) => {
        if (data.column.index === 2 && data.section === 'body') {
          if (data.cell.text[0] === 'Yes') {
            data.cell.styles.fillColor = [34, 197, 94, 0.2] as any;
            data.cell.styles.textColor = REPORT_CONFIG.colors.success;
            data.cell.styles.fontStyle = 'bold';
          } else {
            data.cell.styles.fillColor = [239, 68, 68, 0.2] as any;
            data.cell.styles.textColor = REPORT_CONFIG.colors.danger;
          }
        }
      },
    });

    currentY = doc.lastAutoTable?.finalY || currentY + 60;
  }

  // Recommendations
  doc.addPage();
  currentY = addHeader(doc, 'Model Performance Comparison Report (Continued)');

  currentY = addSectionHeading(doc, '4. Model Selection Recommendations', currentY + 5);

  // Find best model by accuracy
  const bestModel = models.reduce((best, current) =>
    current.accuracy > best.accuracy ? current : best
  );

  currentY = addText(
    doc,
    `Based on the performance metrics, ${bestModel.model_name} demonstrates the highest accuracy (${(bestModel.accuracy * 100).toFixed(2)}%) and is recommended for primary use.`,
    currentY
  );

  currentY += 8;
  currentY = addText(doc, 'Key Considerations:', currentY, { bold: true });
  currentY += 5;

  currentY = addText(
    doc,
    '• Decision Tree: Best for interpretability and understanding decision logic',
    currentY
  );
  currentY += 5;

  currentY = addText(
    doc,
    '• Random Forest: Provides balanced performance with reduced overfitting',
    currentY
  );
  currentY += 5;

  currentY = addText(
    doc,
    '• XGBoost: Often achieves highest accuracy but requires more computational resources',
    currentY
  );
  currentY += 10;

  currentY = addText(
    doc,
    'Recommendation: Use ensemble voting across all models for critical predictions to maximize reliability.',
    currentY,
    { bold: true }
  );

  // Add footers
  const totalPages = doc.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    addFooter(doc, i, totalPages);
  }

  return doc;
}

/**
 * Export Options
 */

/**
 * Download PDF directly
 */
export function downloadPDF(doc: jsPDF, filename: string) {
  doc.save(filename);
}

/**
 * Open PDF in new tab for printing
 */
export function openPDFInNewTab(doc: jsPDF) {
  const pdfBlob = doc.output('blob');
  const blobUrl = URL.createObjectURL(pdfBlob);
  window.open(blobUrl, '_blank');
}

/**
 * Get PDF as blob for uploading/emailing
 */
export function getPDFBlob(doc: jsPDF): Blob {
  return doc.output('blob');
}

/**
 * Get PDF as base64 string
 */
export function getPDFBase64(doc: jsPDF): string {
  return doc.output('dataurlstring');
}

/**
 * Preview PDF (returns data URL for embedding)
 */
export function getPreviewURL(doc: jsPDF): string {
  return doc.output('dataurlstring');
}

/**
 * Send PDF via email (requires backend endpoint)
 */
export async function sendPDFEmail(
  doc: jsPDF,
  emailData: {
    to: string;
    subject: string;
    body: string;
  },
  apiEndpoint: string
): Promise<void> {
  const pdfBlob = getPDFBlob(doc);
  const formData = new FormData();
  formData.append('pdf', pdfBlob, 'report.pdf');
  formData.append('to', emailData.to);
  formData.append('subject', emailData.subject);
  formData.append('body', emailData.body);

  const response = await fetch(apiEndpoint, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Failed to send email');
  }
}
