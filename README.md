# Diabetes Prediction Classification System

A comprehensive machine learning system for predicting diabetes using the Pima Indians Diabetes Dataset. This project includes data exploration, preprocessing, feature engineering, synthetic data generation, model training, and a full-stack web application for predictions.

## Project Overview

This system uses various machine learning algorithms to predict the likelihood of diabetes based on diagnostic measurements. The project implements best practices in data science including:

- Comprehensive Exploratory Data Analysis (EDA)
- Advanced data preprocessing and feature engineering
- Handling class imbalance using SMOTE and variants
- Multiple ML model comparisons (Logistic Regression, Random Forest, XGBoost, etc.)
- Model interpretability using SHAP and LIME
- RESTful API built with FastAPI
- Interactive React frontend for predictions

## Project Structure

```
dm-fp/
├── backend/
│   ├── data/
│   │   ├── raw/              # Original dataset
│   │   ├── processed/        # Preprocessed data
│   │   └── synthetic/        # SMOTE-generated data
│   ├── models/               # Trained model files
│   ├── notebooks/            # Jupyter notebooks for analysis
│   ├── src/                  # Python source code
│   │   ├── preprocessing.py
│   │   ├── synthetic_data_generation.py
│   │   ├── model_training.py
│   │   └── api.py
│   └── results/              # Outputs, graphs, metrics
│       └── eda/              # EDA visualizations
├── frontend/
│   ├── src/                  # React components
│   ├── public/               # Static files
│   └── package.json
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

## Dataset

The Pima Indians Diabetes Dataset contains the following features:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age in years
- **Outcome**: Class variable (0 or 1) - Target

## Installation

### Backend Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Place the diabetes dataset in `backend/data/raw/diabetes.csv`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node dependencies:
```bash
npm install
```

## Usage

### Running Jupyter Notebooks

1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate to `backend/notebooks/` and run the notebooks in order:
   - `01_data_exploration.ipynb` - EDA and visualization
   - Additional notebooks for model training and evaluation

### Running the API

1. Navigate to the backend directory:
```bash
cd backend
```

2. Start the FastAPI server:
```bash
uvicorn src.api:app --reload
```

3. Access the API documentation at `http://localhost:8000/docs`

### Running the Frontend

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Start the development server:
```bash
npm start
```

3. Access the application at `http://localhost:3000`

## Machine Learning Pipeline

### 1. Data Exploration
- Statistical analysis of all features
- Visualization of distributions and correlations
- Outlier detection
- Class imbalance analysis

### 2. Data Preprocessing
- Handling missing values (zeros treated as missing)
- Outlier removal using IQR method
- Feature engineering (BMI categories, glucose levels, age groups)
- Feature scaling using StandardScaler
- Train-test split with stratification

### 3. Synthetic Data Generation
- SMOTE (Synthetic Minority Over-sampling Technique)
- BorderlineSMOTE for borderline samples
- ADASYN (Adaptive Synthetic Sampling)
- Quality validation of synthetic data

### 4. Model Training
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Hyperparameter tuning using GridSearchCV/RandomizedSearchCV

### 5. Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curves
- Confusion matrices
- Cross-validation scores
- Feature importance analysis

### 6. Model Interpretability
- SHAP (SHapley Additive exPlanations) values
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance plots

## API Endpoints

- `POST /predict` - Make diabetes predictions
- `GET /model-info` - Get information about the trained model
- `GET /feature-importance` - Get feature importance scores
- `POST /batch-predict` - Batch predictions for multiple samples

## Technologies Used

### Backend
- **Python 3.8+**
- **FastAPI** - Modern web framework
- **scikit-learn** - Machine learning library
- **XGBoost, LightGBM, CatBoost** - Gradient boosting frameworks
- **pandas, numpy** - Data manipulation
- **matplotlib, seaborn, plotly** - Visualization
- **imbalanced-learn** - SMOTE implementation
- **SHAP, LIME** - Model interpretability

### Frontend
- **React** - UI framework
- **Chart.js** - Data visualization
- **Plotly.js** - Interactive plots
- **Axios** - HTTP client
- **Material-UI** - Component library

## Key Features

- Comprehensive data preprocessing pipeline
- Multiple ML algorithms comparison
- Handling imbalanced datasets
- Model interpretability and explainability
- RESTful API for predictions
- Interactive web interface
- Extensive visualizations and reporting

## A/B Testing Framework

The application includes a comprehensive A/B testing framework for comparing different model versions and optimizing performance.

### Features

- **Traffic Splitting**: Configure traffic percentages for each variant (50/50, 70/30, etc.)
- **User Assignment**: Consistent hashing ensures users see the same variant across sessions
- **Metric Tracking**: Automatically track predictions, conversions, ratings, and interactions
- **Statistical Analysis**: Built-in hypothesis testing with t-tests and confidence intervals
- **Real-time Analytics**: Live dashboards showing experiment progress and results

### Backend API

The A/B testing API provides endpoints for:

**Experiment Management**
- `POST /api/ab-testing/experiments` - Create new experiment
- `GET /api/ab-testing/experiments` - List all experiments
- `GET /api/ab-testing/experiments/{id}` - Get experiment details
- `POST /api/ab-testing/experiments/{id}/start` - Start experiment
- `POST /api/ab-testing/experiments/{id}/pause` - Pause experiment
- `POST /api/ab-testing/experiments/{id}/stop` - Stop experiment

**User Assignment**
- `GET /api/ab-testing/experiments/{id}/assign` - Assign user to variant
- `GET /api/ab-testing/experiments/{id}/variant` - Get user's assigned variant

**Tracking**
- `POST /api/ab-testing/track/prediction` - Track prediction event
- `POST /api/ab-testing/track/conversion` - Track conversion event
- `POST /api/ab-testing/track/rating` - Track user rating
- `POST /api/ab-testing/track/interaction` - Track user interaction

**Analytics**
- `GET /api/ab-testing/experiments/{id}/results` - Get comprehensive results
- `GET /api/ab-testing/experiments/{id}/metrics` - Get current metrics
- `POST /api/ab-testing/experiments/{id}/analyze` - Run statistical analysis

### Frontend Integration

#### Using the A/B Test Hook

```typescript
import { useABTest } from '@/hooks/useABTest';

function MyComponent() {
  const abTest = useABTest('experiment-id', {
    enabled: true,
    autoAssign: true
  });

  // Track prediction
  abTest.trackPrediction({
    prediction_time_ms: 145,
    confidence: 0.87,
    prediction: 1,
    risk_level: 'high',
  });

  // Track conversion
  abTest.trackConversion(true);

  // Track rating
  abTest.trackRating(4.5);

  return (
    <div>
      <p>Using model: {abTest.modelName}</p>
      <p>Variant: {abTest.variantId}</p>
    </div>
  );
}
```

#### Admin Dashboard

View and manage experiments:
```typescript
import ABTestAdmin from '@/components/ABTestAdmin';

function AdminPage() {
  return <ABTestAdmin onViewExperiment={(id) => navigate(`/experiments/${id}`)} />;
}
```

#### Analytics Dashboard

View experiment results:
```typescript
import ABTestDashboard from '@/components/ABTestDashboard';

function ResultsPage() {
  return <ABTestDashboard experimentId="experiment-id" />;
}
```

### Creating an Experiment

```bash
curl -X POST http://localhost:8000/api/ab-testing/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "name": "RF vs XGBoost",
    "description": "Compare Random Forest and XGBoost models",
    "variants": [
      {
        "name": "Control (RF)",
        "model_name": "random_forest",
        "traffic_percentage": 50,
        "variant_type": "control"
      },
      {
        "name": "Treatment (XGB)",
        "model_name": "xgboost",
        "traffic_percentage": 50,
        "variant_type": "treatment"
      }
    ],
    "target_metric": "conversion_rate",
    "min_sample_size": 100,
    "confidence_level": 0.95
  }'
```

### Metrics Tracked

- **Sample Size**: Total users and predictions per variant
- **Performance**: Avg prediction time, confidence scores
- **Predictions**: Distribution of positive/negative and risk levels
- **Engagement**: User interactions, conversion rate
- **Satisfaction**: User ratings (1-5 stars)
- **Errors**: Error count and rate

### Statistical Analysis

The framework uses two-sample t-tests to determine statistical significance:

- **Null Hypothesis**: No difference between variants
- **Confidence Level**: 95% (configurable)
- **Min Sample Size**: 100 (configurable)
- **Metrics**: Relative lift, p-value, confidence intervals
- **Recommendations**: Automatic suggestions based on results

### Example Results

```json
{
  "winner": {
    "winner": "variant-123",
    "variant_name": "Treatment (XGB)",
    "lift": 12.5,
    "confidence": 0.95
  },
  "comparisons": [{
    "significant": true,
    "p_value": 0.023,
    "relative_lift_percent": 12.5,
    "control_mean": 0.72,
    "treatment_mean": 0.81,
    "recommendation": "Strong positive lift detected! Consider promoting treatment variant."
  }]
}
```

## Future Enhancements

- Deploy to cloud (AWS/GCP/Azure)
- Add user authentication
- Implement model monitoring and retraining pipeline
- Add more advanced ensemble methods
- Include deep learning models
- Expand dataset with additional features
- Mobile application
- Multi-armed bandit algorithms for A/B testing
- Bayesian A/B testing methods

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Pima Indians Diabetes Dataset from Kaggle
- National Institute of Diabetes and Digestive and Kidney Diseases

## Contact

For questions or feedback, please open an issue in the repository.
