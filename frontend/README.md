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
- **Batch Analysis**: Analyze multiple patients at once
- **Model Comparison**: Compare predictions from all ML models
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
