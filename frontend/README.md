# Diabetes Prediction Frontend

React + TypeScript frontend for the Diabetes Prediction ML System.

## Features

- **Modern Tech Stack**: Built with Vite, React 18, and TypeScript
- **Responsive Design**: Mobile-first design with Tailwind CSS
- **Dark Mode**: Full dark mode support with theme toggle
- **API Integration**: Complete integration with FastAPI backend
- **Interactive Charts**: Visualizations using Chart.js, Plotly, and Recharts
- **Type Safety**: Full TypeScript coverage for better developer experience

## Pages

- **Dashboard**: Overview of system status, models, and statistics
- **Single Prediction**: Predict diabetes risk for one patient with SHAP explanations
  - **Results & Insights**: Comprehensive analysis with visualizations
  - **Interactive Explorer**: Real-time feature exploration and what-if scenarios
- **Batch Analysis**: Upload CSV files to analyze multiple patients at once
  - CSV file upload with drag-and-drop
  - Data preview before processing
  - Batch processing with progress tracking
  - Statistics dashboard with risk distribution charts
  - Sortable and filterable results table
  - Detailed patient view with comprehensive analysis
  - Export results as CSV or summary report
- **Model Comparison**: Compare predictions from all ML models
- **Visualization Dashboard**: Interactive data visualizations and statistical analysis
  - Feature distribution histograms and box plots
  - Correlation heatmap with interactive hover
  - 3D scatter plots with rotate and zoom
  - Pairplot matrix for multi-feature analysis
  - Summary statistics cards
  - Customizable color schemes and filters
- **Model Explainability**: Educational page explaining how ML models make decisions
  - Learn how Decision Trees, Random Forests, and XGBoost work
  - Interactive decision tree visualization
  - Feature importance comparison across all models
  - SHAP values explanation with waterfall plots
  - Example predictions with step-by-step explanations
  - Try-it-yourself section with real-time model comparisons
- **About Models**: Learn about Decision Tree, Random Forest, and XGBoost

## Prerequisites

- Node.js 16+ and npm
- Backend API running on http://localhost:8000

## Installation

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create environment file:
```bash
cp .env.example .env
```

4. Update `.env` with your API base URL (default: http://localhost:8000)

## Development

Start development server:
```bash
npm run dev
```

The app will be available at http://localhost:3000

## Building for Production

Build the app:
```bash
npm run build
```

Preview production build:
```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── common/          # Reusable components
│   │   │   ├── Button.tsx
│   │   │   ├── Card.tsx
│   │   │   ├── ErrorMessage.tsx
│   │   │   └── LoadingSpinner.tsx
│   │   └── layout/          # Layout components
│   │       ├── Header.tsx
│   │       ├── Sidebar.tsx
│   │       └── MainLayout.tsx
│   ├── pages/               # Page components
│   │   ├── Dashboard.tsx
│   │   ├── SinglePrediction.tsx
│   │   ├── BatchAnalysis.tsx
│   │   ├── ModelComparison.tsx
│   │   ├── VisualizationDashboard.tsx
│   │   ├── ModelExplainability.tsx
│   │   └── About.tsx
│   ├── services/            # API services
│   │   └── api.ts
│   ├── types/               # TypeScript types
│   │   └── api.ts
│   ├── App.tsx              # Main app component
│   ├── main.tsx             # Entry point
│   └── index.css            # Global styles
├── public/                  # Static assets
├── index.html               # HTML template
├── vite.config.ts           # Vite configuration
├── tailwind.config.js       # Tailwind configuration
├── tsconfig.json            # TypeScript configuration
└── package.json             # Dependencies
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Lint code with ESLint

## Components

### Feature Components

#### FeatureExplorer
Interactive component for real-time feature exploration and what-if analysis:

```tsx
<FeatureExplorer initialData={patientData} />
```

**Features:**
- **Real-time Predictions**: Adjusting sliders triggers debounced predictions (500ms delay)
- **Risk Meter**: Visual gauge showing current diabetes risk percentage
- **What-if Scenarios**: Quick action buttons (lose weight, lower glucose, younger age)
- **Feature Analysis**: Partial dependence plots showing how individual features affect risk
- **Scenario Comparison**: Save up to 3 scenarios and compare them side-by-side
- **Interactive Sliders**: All 8 patient features with visual feedback
- **Animated Transitions**: Smooth color changes and SVG animations (700ms duration)

**Props:**
- `initialData`: Optional `PatientInput` object to pre-populate feature values

#### BatchAnalysis
Complete batch prediction page with file upload and analysis:

**Features:**
- **File Upload**: Drag-and-drop CSV file upload with validation using react-dropzone
- **CSV Parsing**: Automatic parsing with papaparse and column validation
- **Preview**: View first 5 rows before processing
- **Batch Processing**: Process up to 100s of patients with progress tracking (10 patients per batch)
- **Statistics Dashboard**:
  - Total patients, risk distribution (Low/Medium/High)
  - Average risk probability
  - Pie chart for risk distribution (Chart.js)
  - Histogram for probability distribution (Chart.js)
  - Top 5 highest risk patients with quick view
- **Results Table**:
  - Sortable columns (Age, Glucose, BMI, Probability, Risk Level)
  - Search functionality (by ID or risk level)
  - Filter by risk level (All/Low/Medium/High)
  - Color-coded risk badges
  - Row hover effects and smooth interactions
- **Detailed Patient View**:
  - Full-screen modal with PredictionResults component
  - On-demand fetching of detailed predictions with SHAP explanations
  - Smooth loading states
- **Export Functionality**:
  - Export results as CSV with predictions and confidence scores
  - Export summary report as text file with recommendations
  - Download individual patient analysis

**Required CSV Format:**
```csv
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
6,148,72,35,0,33.6,0.627,50
1,85,66,29,0,26.6,0.351,31
```

**Dependencies:**
- `react-dropzone`: File upload component
- `papaparse`: CSV parsing library

#### VisualizationDashboard
Comprehensive data visualization dashboard with interactive charts:

**Features:**
- **Summary Statistics Cards**: 5 metric cards showing total patients, diabetic percentage, average age/BMI, and feature count
- **Feature Distribution Analysis**:
  - Dual histograms comparing diabetic vs non-diabetic distributions
  - Side-by-side box plots showing quartiles and outliers
  - Summary statistics table (mean, std dev, min, median, max)
  - Feature selector dropdown
- **Correlation Heatmap**:
  - Interactive heatmap showing all feature correlations
  - Hover to see exact correlation values
  - Customizable color schemes (Viridis, Plasma, Cool, Warm, Earth)
  - Plotly-based with zoom and pan
- **3D Scatter Plot**:
  - Interactive 3D visualization with rotate and zoom
  - Separate colors for diabetic (red) and non-diabetic (green) patients
  - Configurable X, Y, Z axes from any feature
  - Plotly 3D scatter with camera controls
- **Pairplot Matrix**:
  - 3x3 grid of scatter plots and distributions
  - Diagonal shows feature distributions
  - Off-diagonal shows pairwise scatter plots
  - Select any 3 features to visualize
- **Global Controls**:
  - Color scheme selector (5 schemes)
  - Outcome filter (All/Diabetic/Non-Diabetic)
  - Download buttons for all charts
  - Refresh data button
- **Interactive Features**:
  - Hover tooltips on all charts
  - Click legend items to toggle series
  - Responsive layouts for all screen sizes
  - Dark mode support with transparent backgrounds

**Chart Types:**
- Plotly.js for advanced visualizations (heatmap, 3D, interactive plots)
- Multiple color schemes for customization
- Export functionality for all charts

#### ModelExplainability
Educational page explaining how ML models work and make predictions:

**Features:**
- **Education Section**:
  - Comprehensive explanations for Decision Tree, Random Forest, and XGBoost
  - How each model works (step-by-step breakdown)
  - Pros and cons for each model
  - When to use each model
  - Interactive model selector with tabbed interface
- **Decision Tree Visualization**:
  - Simplified HTML/CSS-based tree structure
  - Color-coded nodes (blue for decisions, green/red for outcomes)
  - Visual representation of decision-making process
  - Explanation of tree traversal
- **Feature Importance Comparison**:
  - Side-by-side comparison of all 3 models
  - Interactive Plotly grouped bar chart
  - Educational cards explaining:
    - What feature importance means
    - Why different models rank features differently
  - Sorted by importance with clear visual hierarchy
- **SHAP Values Section**:
  - Introduction to SHAP (SHapley Additive exPlanations)
  - How to interpret SHAP values (positive vs negative contributions)
  - Example SHAP waterfall plot showing feature contributions
  - Color-coded interpretation guide
- **Example Predictions**:
  - 3 sample patients (Low Risk, High Risk, Borderline)
  - Loads real predictions from all models on page mount
  - Side-by-side comparison showing how models agree/disagree
  - Patient data displayed with expected outcomes
- **Try It Yourself**:
  - Interactive form with all 8 patient feature inputs
  - Real-time predictions from all models
  - Comprehensive prediction with SHAP explanations
  - Visual comparison of model reasoning
  - Feature contribution waterfall plot

**Navigation:**
- 6 section tabs with icons (Learn, Trees, Importance, SHAP, Examples, Try It)
- Smooth scrolling and progressive disclosure
- Mobile-responsive with proper spacing

**Dependencies:**
- Plotly.js for interactive charts
- React Icons (react-icons/fi) for section icons
- Integrates with `/api/predict-explain` and `/api/compare-models` endpoints

#### PredictionForm
Comprehensive input form with dual inputs (sliders + number fields):

```tsx
<PredictionForm
  onPredictionComplete={(result) => console.log(result)}
  onPredictionStart={() => console.log('Started')}
/>
```

**Features:**
- Dual input controls (range slider + number input)
- Real-time validation with error messages
- Visual range indicators (normal ranges highlighted)
- Color coding (green/yellow/red based on risk)
- Tooltips with feature descriptions
- Example patient presets (high/low/moderate risk)
- Random patient generator

#### PredictionResults
Multi-section results dashboard with visualizations:

```tsx
<PredictionResults result={comprehensivePrediction} />
```

**Sections:**
- Risk assessment with circular progress (Chart.js doughnut)
- Model predictions comparison table
- Feature importance horizontal bar chart
- SHAP waterfall explanation (Plotly)
- Risk factors grid with modifiable/non-modifiable indicators
- Personalized recommendations with priority levels
- Similar patients comparison
- Export functionality (PDF, email, local storage)

### Common Components

#### Button
```tsx
<Button variant="primary" size="md" onClick={handleClick}>
  Click Me
</Button>
```

Variants: `primary`, `secondary`, `success`, `danger`, `outline`
Sizes: `sm`, `md`, `lg`

#### Card
```tsx
<Card title="My Card" subtitle="Description">
  Content goes here
</Card>
```

#### LoadingSpinner
```tsx
<LoadingSpinner size="md" text="Loading..." />
```

#### ErrorMessage
```tsx
<ErrorMessage
  title="Error"
  message="Something went wrong"
  variant="error"
/>
```

## API Service

The API service provides functions for all backend endpoints:

```typescript
import { predictionApi, modelApi, dataApi, healthApi } from '@/services/api';

// Health check
const health = await healthApi.check();

// Single prediction
const result = await predictionApi.predict(patientData, 'xgboost');

// Comprehensive prediction with explanations
const comprehensive = await predictionApi.predictExplain(patientData);

// List models
const models = await modelApi.listModels();

// Get feature importance
const features = await modelApi.getFeatureImportance('xgboost', 10);

// Dataset statistics
const stats = await dataApi.getStats();
```

## Styling

### Tailwind CSS

The project uses Tailwind CSS with custom configuration:

- Custom color palette (primary, success, warning, danger)
- Dark mode support
- Custom utility classes
- Responsive breakpoints

### Custom Classes

```css
.btn - Base button styles
.btn-primary - Primary button
.btn-secondary - Secondary button
.card - Card container
.input - Form input
.label - Form label
```

## Theme Toggle

The app supports dark mode with automatic detection of system preference:

```typescript
// Theme is stored in localStorage
// Toggle with the theme button in the header
```

## TypeScript

All components and services are fully typed:

```typescript
import type { PatientInput, PredictionOutput } from '@/types/api';
```

## Environment Variables

- `VITE_API_BASE_URL` - Backend API base URL (default: http://localhost:8000)

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Performance

- Code splitting with React Router
- Lazy loading for routes
- Optimized bundle size with Vite
- Fast HMR in development

## Troubleshooting

### API Connection Issues

1. Ensure backend is running on http://localhost:8000
2. Check CORS configuration in backend
3. Verify API endpoints are accessible

### Build Issues

```bash
# Clear node_modules and reinstall
rm -rf node_modules
npm install

# Clear Vite cache
rm -rf node_modules/.vite
```

## Contributing

1. Follow TypeScript best practices
2. Use functional components with hooks
3. Maintain component modularity
4. Add proper TypeScript types
5. Follow existing code style

## License

MIT License
