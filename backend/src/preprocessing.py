"""
Data Preprocessing Module for Diabetes Prediction

This module contains the DataPreprocessor class which handles all data preprocessing tasks including:
- Handling missing values
- Outlier detection and removal
- Feature engineering
- Feature scaling
- Train-test splitting

Author: Diabetes Prediction Project
Date: 2025
"""

import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A comprehensive data preprocessing class for the diabetes dataset.

    This class provides methods for handling missing values, removing outliers,
    creating new features, scaling features, and splitting data.
    """

    def __init__(self, data_path=None, df=None):
        """
        Initialize the DataPreprocessor.

        Args:
            data_path (str, optional): Path to the CSV file containing the dataset
            df (pd.DataFrame, optional): DataFrame to preprocess
        """
        if df is not None:
            self.df = df.copy()
            logger.info("DataFrame loaded successfully")
        elif data_path is not None:
            try:
                self.df = pd.read_csv(data_path)
                logger.info(f"Dataset loaded from {data_path}")
                logger.info(f"Dataset shape: {self.df.shape}")
            except Exception as e:
                logger.error(f"Error loading dataset: {str(e)}")
                raise
        else:
            raise ValueError("Either data_path or df must be provided")

        self.scaler = StandardScaler()
        self.feature_columns = None
        self.original_shape = self.df.shape

    def handle_missing_values(self, method='median'):
        """
        Handle missing values in the dataset.

        In the diabetes dataset, zeros in certain columns (Glucose, BloodPressure,
        BMI, Insulin, SkinThickness) are biologically impossible and represent missing data.

        Args:
            method (str): Method to fill missing values ('median', 'mean', or 'drop')

        Returns:
            self: Returns self for method chaining
        """
        try:
            logger.info("Handling missing values...")

            # Columns where 0 represents missing data
            zero_as_missing_cols = ['Glucose', 'BloodPressure', 'BMI', 'Insulin', 'SkinThickness']

            # Log missing value counts before handling
            missing_counts = {}
            for col in zero_as_missing_cols:
                if col in self.df.columns:
                    count = (self.df[col] == 0).sum()
                    missing_counts[col] = count
                    if count > 0:
                        logger.info(f"{col}: {count} zero values detected")

            # Replace zeros with NaN
            for col in zero_as_missing_cols:
                if col in self.df.columns:
                    self.df[col] = self.df[col].replace(0, np.nan)

            # Fill missing values based on method
            if method == 'median':
                for col in zero_as_missing_cols:
                    if col in self.df.columns:
                        median_val = self.df[col].median()
                        self.df[col].fillna(median_val, inplace=True)
                        logger.info(f"{col}: Filled with median ({median_val:.2f})")

            elif method == 'mean':
                for col in zero_as_missing_cols:
                    if col in self.df.columns:
                        mean_val = self.df[col].mean()
                        self.df[col].fillna(mean_val, inplace=True)
                        logger.info(f"{col}: Filled with mean ({mean_val:.2f})")

            elif method == 'drop':
                initial_rows = len(self.df)
                self.df.dropna(subset=zero_as_missing_cols, inplace=True)
                rows_dropped = initial_rows - len(self.df)
                logger.info(f"Dropped {rows_dropped} rows with missing values")

            else:
                raise ValueError("Method must be 'median', 'mean', or 'drop'")

            logger.info(f"Missing value handling complete. Dataset shape: {self.df.shape}")
            return self

        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise

    def remove_outliers(self, method='iqr', multiplier=1.5):
        """
        Remove outliers from the dataset using the IQR method.

        Args:
            method (str): Method for outlier removal ('iqr' or 'zscore')
            multiplier (float): IQR multiplier for outlier detection (default: 1.5)

        Returns:
            self: Returns self for method chaining
        """
        try:
            logger.info(f"Removing outliers using {method} method...")
            initial_rows = len(self.df)

            # Get numerical columns (excluding Outcome)
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Outcome' in numerical_cols:
                numerical_cols.remove('Outcome')

            if method == 'iqr':
                # IQR method
                for col in numerical_cols:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - multiplier * IQR
                    upper_bound = Q3 + multiplier * IQR

                    # Count outliers before removal
                    outliers_before = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()

                    # Remove outliers
                    self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]

                    if outliers_before > 0:
                        logger.info(f"{col}: Removed {outliers_before} outliers (bounds: {lower_bound:.2f} - {upper_bound:.2f})")

            elif method == 'zscore':
                # Z-score method
                from scipy import stats
                z_scores = np.abs(stats.zscore(self.df[numerical_cols]))
                self.df = self.df[(z_scores < 3).all(axis=1)]

            else:
                raise ValueError("Method must be 'iqr' or 'zscore'")

            rows_removed = initial_rows - len(self.df)
            logger.info(f"Outlier removal complete. Removed {rows_removed} rows. New shape: {self.df.shape}")
            return self

        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")
            raise

    def create_features(self):
        """
        Engineer new features from existing ones.

        Creates:
        - BMI_Category: Categorical BMI classification
        - Glucose_Category: Categorical glucose level classification
        - Age_Group: Age group classification
        - Risk_Score: Composite risk score
        - Interaction features

        Returns:
            self: Returns self for method chaining
        """
        try:
            logger.info("Creating engineered features...")

            # 1. BMI Categories
            def categorize_bmi(bmi):
                if bmi < 18.5:
                    return 0  # Underweight
                elif 18.5 <= bmi < 25:
                    return 1  # Normal
                elif 25 <= bmi < 30:
                    return 2  # Overweight
                else:
                    return 3  # Obese

            self.df['BMI_Category'] = self.df['BMI'].apply(categorize_bmi)
            logger.info("Created BMI_Category feature")

            # 2. Glucose Categories
            def categorize_glucose(glucose):
                if glucose < 100:
                    return 0  # Normal
                elif 100 <= glucose < 126:
                    return 1  # Prediabetic
                else:
                    return 2  # Diabetic

            self.df['Glucose_Category'] = self.df['Glucose'].apply(categorize_glucose)
            logger.info("Created Glucose_Category feature")

            # 3. Age Groups
            def categorize_age(age):
                if age < 30:
                    return 0  # Young
                elif 30 <= age < 50:
                    return 1  # Middle-aged
                else:
                    return 2  # Senior

            self.df['Age_Group'] = self.df['Age'].apply(categorize_age)
            logger.info("Created Age_Group feature")

            # 4. Risk Score (composite feature)
            # Normalize features to 0-1 scale for risk score calculation
            def normalize(series):
                return (series - series.min()) / (series.max() - series.min())

            self.df['Risk_Score'] = (
                0.3 * normalize(self.df['Glucose']) +
                0.25 * normalize(self.df['BMI']) +
                0.15 * normalize(self.df['Age']) +
                0.15 * normalize(self.df['DiabetesPedigreeFunction']) +
                0.15 * normalize(self.df['Pregnancies'])
            )
            logger.info("Created Risk_Score feature")

            # 5. Interaction features
            self.df['Glucose_BMI_Interaction'] = self.df['Glucose'] * self.df['BMI']
            self.df['Age_Glucose_Interaction'] = self.df['Age'] * self.df['Glucose']
            logger.info("Created interaction features")

            # 6. Binary features
            self.df['High_Glucose'] = (self.df['Glucose'] >= 126).astype(int)
            self.df['High_BMI'] = (self.df['BMI'] >= 30).astype(int)
            self.df['High_BP'] = (self.df['BloodPressure'] >= 80).astype(int)
            logger.info("Created binary indicator features")

            logger.info(f"Feature engineering complete. New shape: {self.df.shape}")
            return self

        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            raise

    def scale_features(self, exclude_cols=None):
        """
        Scale numerical features using StandardScaler.

        Args:
            exclude_cols (list): Columns to exclude from scaling (e.g., target variable)

        Returns:
            self: Returns self for method chaining
        """
        try:
            logger.info("Scaling features...")

            if exclude_cols is None:
                exclude_cols = ['Outcome']

            # Get columns to scale
            cols_to_scale = [col for col in self.df.columns if col not in exclude_cols]

            # Save feature columns for later use
            self.feature_columns = cols_to_scale

            # Fit and transform
            self.df[cols_to_scale] = self.scaler.fit_transform(self.df[cols_to_scale])

            logger.info(f"Scaled {len(cols_to_scale)} features")
            logger.info(f"Feature columns: {cols_to_scale}")
            return self

        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise

    def train_test_split_data(self, test_size=0.2, random_state=42, stratify=True):
        """
        Split the data into training and testing sets.

        Args:
            test_size (float): Proportion of data for testing (default: 0.2)
            random_state (int): Random seed for reproducibility (default: 42)
            stratify (bool): Whether to stratify split based on target variable

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        try:
            logger.info("Splitting data into train and test sets...")

            # Separate features and target
            X = self.df.drop('Outcome', axis=1)
            y = self.df['Outcome']

            # Split data
            stratify_col = y if stratify else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_col
            )

            logger.info(f"Train set size: {X_train.shape}")
            logger.info(f"Test set size: {X_test.shape}")
            logger.info(f"Train set class distribution:\n{y_train.value_counts()}")
            logger.info(f"Test set class distribution:\n{y_test.value_counts()}")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise

    def save_preprocessed_data(self, X_train, X_test, y_train, y_test, output_dir='../data/processed'):
        """
        Save preprocessed data and scaler to disk.

        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
            output_dir (str): Directory to save the files
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            logger.info(f"Saving preprocessed data to {output_dir}...")

            # Save datasets
            X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
            X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
            y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
            y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

            # Save scaler
            joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))

            # Save feature columns
            feature_info = {
                'feature_columns': self.feature_columns if self.feature_columns else list(X_train.columns),
                'n_features': X_train.shape[1]
            }
            joblib.dump(feature_info, os.path.join(output_dir, 'feature_info.pkl'))

            logger.info("All files saved successfully!")
            logger.info(f"Files saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv, scaler.pkl, feature_info.pkl")

        except Exception as e:
            logger.error(f"Error saving preprocessed data: {str(e)}")
            raise

    def get_preprocessing_summary(self):
        """
        Get a summary of the preprocessing steps applied.

        Returns:
            dict: Summary statistics and information
        """
        summary = {
            'original_shape': self.original_shape,
            'current_shape': self.df.shape,
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'features_created': self.df.shape[1] - self.original_shape[1],
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return summary


def preprocess_pipeline(data_path, output_dir='../data/processed',
                        remove_outliers_flag=True, create_features_flag=True):
    """
    Complete preprocessing pipeline.

    Args:
        data_path (str): Path to the raw dataset
        output_dir (str): Directory to save preprocessed data
        remove_outliers_flag (bool): Whether to remove outliers
        create_features_flag (bool): Whether to create engineered features

    Returns:
        tuple: X_train, X_test, y_train, y_test, preprocessor
    """
    try:
        logger.info("="*80)
        logger.info("STARTING PREPROCESSING PIPELINE")
        logger.info("="*80)

        # Initialize preprocessor
        preprocessor = DataPreprocessor(data_path=data_path)

        # Step 1: Handle missing values
        preprocessor.handle_missing_values(method='median')

        # Step 2: Remove outliers (optional)
        if remove_outliers_flag:
            preprocessor.remove_outliers(method='iqr', multiplier=1.5)

        # Step 3: Create features (optional)
        if create_features_flag:
            preprocessor.create_features()

        # Step 4: Split data before scaling
        X_train, X_test, y_train, y_test = preprocessor.train_test_split_data(
            test_size=0.2, random_state=42, stratify=True
        )

        # Step 5: Scale features
        # Combine train and test for scaling, then split again
        full_data = pd.concat([X_train, X_test])
        full_target = pd.concat([y_train, y_test])
        full_data['Outcome'] = full_target

        preprocessor.df = full_data
        preprocessor.scale_features(exclude_cols=['Outcome'])

        # Split again after scaling
        X_train, X_test, y_train, y_test = preprocessor.train_test_split_data(
            test_size=0.2, random_state=42, stratify=True
        )

        # Step 6: Save preprocessed data
        preprocessor.save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir)

        # Print summary
        logger.info("="*80)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("="*80)
        summary = preprocessor.get_preprocessing_summary()
        for key, value in summary.items():
            if key not in ['columns', 'missing_values', 'data_types']:
                logger.info(f"{key}: {value}")

        logger.info("="*80)
        logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)

        return X_train, X_test, y_train, y_test, preprocessor

    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    print("Diabetes Data Preprocessing Module")
    print("="*80)
    print("\nThis module provides comprehensive data preprocessing capabilities.")
    print("\nUsage example:")
    print("""
    from preprocessing import preprocess_pipeline

    # Run complete preprocessing pipeline
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(
        data_path='../data/raw/diabetes.csv',
        output_dir='../data/processed',
        remove_outliers_flag=True,
        create_features_flag=True
    )
    """)
