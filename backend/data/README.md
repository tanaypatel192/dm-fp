# Data Directory

This directory contains all datasets used in the diabetes prediction project.

## Structure

- **raw/**: Original, unprocessed datasets
  - Place your `diabetes.csv` file here

- **processed/**: Preprocessed datasets ready for model training
  - X_train.csv
  - X_test.csv
  - y_train.csv
  - y_test.csv
  - scaler.pkl
  - feature_info.pkl

- **synthetic/**: Datasets generated using SMOTE and other techniques
  - X_train_resampled.csv
  - y_train_resampled.csv

## Dataset Source

The Pima Indians Diabetes Dataset can be downloaded from:
- Kaggle: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/diabetes

## Usage

1. Download the diabetes.csv file
2. Place it in the `raw/` directory
3. Run the preprocessing pipeline
4. Processed files will be saved to `processed/` directory
5. Run SMOTE to generate synthetic data in `synthetic/` directory
