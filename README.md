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

## Future Enhancements

- Deploy to cloud (AWS/GCP/Azure)
- Add user authentication
- Implement model monitoring and retraining pipeline
- Add more advanced ensemble methods
- Include deep learning models
- Expand dataset with additional features
- Mobile application

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Pima Indians Diabetes Dataset from Kaggle
- National Institute of Diabetes and Digestive and Kidney Diseases

## Contact

For questions or feedback, please open an issue in the repository.
