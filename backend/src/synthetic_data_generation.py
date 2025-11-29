"""
Synthetic Data Generation Module for Diabetes Prediction

This module handles class imbalance using various synthetic data generation techniques:
- SMOTE (Synthetic Minority Over-sampling Technique)
- BorderlineSMOTE
- ADASYN (Adaptive Synthetic Sampling)

It also includes validation methods to ensure synthetic data quality.

Author: Diabetes Prediction Project
Date: 2025
"""

import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Imbalanced-learn imports
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    A class for generating synthetic data to handle class imbalance.

    Supports multiple sampling strategies including SMOTE, BorderlineSMOTE,
    and ADASYN with quality validation.
    """

    def __init__(self, random_state=42):
        """
        Initialize the SyntheticDataGenerator.

        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.sampler = None
        self.sampling_strategy = None
        self.original_distribution = None
        self.new_distribution = None

    def generate_synthetic_data(self, X_train, y_train, method='smote',
                                sampling_strategy='auto', k_neighbors=5):
        """
        Generate synthetic data using the specified method.

        Args:
            X_train: Training features
            y_train: Training labels
            method (str): Sampling method ('smote', 'borderline', 'adasyn', 'svmsmote')
            sampling_strategy: Sampling strategy ('auto', 'minority', or dict/float)
            k_neighbors (int): Number of nearest neighbors for SMOTE

        Returns:
            tuple: X_resampled, y_resampled
        """
        try:
            logger.info("="*80)
            logger.info(f"GENERATING SYNTHETIC DATA USING {method.upper()}")
            logger.info("="*80)

            # Log original class distribution
            self.original_distribution = Counter(y_train)
            logger.info(f"Original class distribution: {dict(self.original_distribution)}")
            logger.info(f"Original dataset size: {len(y_train)}")

            # Select sampling method
            if method.lower() == 'smote':
                self.sampler = SMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=k_neighbors,
                    random_state=self.random_state
                )
                logger.info(f"Using SMOTE with k_neighbors={k_neighbors}")

            elif method.lower() == 'borderline':
                self.sampler = BorderlineSMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=k_neighbors,
                    random_state=self.random_state
                )
                logger.info(f"Using BorderlineSMOTE with k_neighbors={k_neighbors}")

            elif method.lower() == 'adasyn':
                self.sampler = ADASYN(
                    sampling_strategy=sampling_strategy,
                    n_neighbors=k_neighbors,
                    random_state=self.random_state
                )
                logger.info(f"Using ADASYN with n_neighbors={k_neighbors}")

            elif method.lower() == 'svmsmote':
                self.sampler = SVMSMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=k_neighbors,
                    random_state=self.random_state
                )
                logger.info(f"Using SVMSMOTE with k_neighbors={k_neighbors}")

            else:
                raise ValueError(f"Unknown method: {method}. Choose from 'smote', 'borderline', 'adasyn', 'svmsmote'")

            # Apply resampling
            X_resampled, y_resampled = self.sampler.fit_resample(X_train, y_train)

            # Log new class distribution
            self.new_distribution = Counter(y_resampled)
            logger.info(f"New class distribution: {dict(self.new_distribution)}")
            logger.info(f"New dataset size: {len(y_resampled)}")

            # Calculate statistics
            synthetic_samples = len(y_resampled) - len(y_train)
            logger.info(f"Synthetic samples generated: {synthetic_samples}")
            logger.info(f"Dataset size increase: {(len(y_resampled)/len(y_train) - 1)*100:.2f}%")

            logger.info("="*80)
            logger.info("SYNTHETIC DATA GENERATION COMPLETED")
            logger.info("="*80)

            return X_resampled, y_resampled

        except Exception as e:
            logger.error(f"Error generating synthetic data: {str(e)}")
            raise

    def validate_synthetic_data(self, X_train, y_train, X_resampled, y_resampled,
                                feature_names=None):
        """
        Validate the quality of synthetic data using statistical tests.

        Args:
            X_train: Original training features
            y_train: Original training labels
            X_resampled: Resampled features (with synthetic data)
            y_resampled: Resampled labels
            feature_names (list): Names of features

        Returns:
            dict: Validation results
        """
        try:
            logger.info("Validating synthetic data quality...")

            if feature_names is None:
                if isinstance(X_train, pd.DataFrame):
                    feature_names = X_train.columns.tolist()
                else:
                    feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

            validation_results = {
                'ks_test_results': [],
                'mean_comparison': [],
                'std_comparison': []
            }

            # Convert to numpy arrays if needed
            if isinstance(X_train, pd.DataFrame):
                X_train_arr = X_train.values
                X_resampled_arr = X_resampled.values
            else:
                X_train_arr = X_train
                X_resampled_arr = X_resampled

            # For minority class samples only
            minority_class = min(self.original_distribution.keys(),
                               key=lambda x: self.original_distribution[x])

            original_minority = X_train_arr[y_train == minority_class]
            resampled_minority = X_resampled_arr[y_resampled == minority_class]

            logger.info(f"Validating synthetic data for minority class: {minority_class}")
            logger.info("="*80)

            # Kolmogorov-Smirnov test for each feature
            for i, feature_name in enumerate(feature_names):
                # KS test
                ks_stat, p_value = stats.ks_2samp(
                    original_minority[:, i],
                    resampled_minority[:, i]
                )

                # Mean and std comparison
                orig_mean = original_minority[:, i].mean()
                resamp_mean = resampled_minority[:, i].mean()
                orig_std = original_minority[:, i].std()
                resamp_std = resampled_minority[:, i].std()

                validation_results['ks_test_results'].append({
                    'feature': feature_name,
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'similar': p_value > 0.05
                })

                validation_results['mean_comparison'].append({
                    'feature': feature_name,
                    'original_mean': orig_mean,
                    'resampled_mean': resamp_mean,
                    'difference': abs(orig_mean - resamp_mean)
                })

                validation_results['std_comparison'].append({
                    'feature': feature_name,
                    'original_std': orig_std,
                    'resampled_std': resamp_std,
                    'difference': abs(orig_std - resamp_std)
                })

                logger.info(f"{feature_name}:")
                logger.info(f"  KS Statistic: {ks_stat:.4f}, p-value: {p_value:.4f} "
                          f"({'Similar' if p_value > 0.05 else 'Different'})")
                logger.info(f"  Mean - Original: {orig_mean:.4f}, Resampled: {resamp_mean:.4f}")
                logger.info(f"  Std  - Original: {orig_std:.4f}, Resampled: {resamp_std:.4f}")

            # Overall similarity score
            similar_features = sum(1 for r in validation_results['ks_test_results'] if r['similar'])
            similarity_percentage = (similar_features / len(feature_names)) * 100

            logger.info("="*80)
            logger.info(f"Validation Summary:")
            logger.info(f"Features with similar distributions: {similar_features}/{len(feature_names)} "
                      f"({similarity_percentage:.2f}%)")
            logger.info("="*80)

            validation_results['similarity_percentage'] = similarity_percentage

            return validation_results

        except Exception as e:
            logger.error(f"Error validating synthetic data: {str(e)}")
            raise

    def visualize_comparison(self, X_train, y_train, X_resampled, y_resampled,
                            output_dir='../results/eda', feature_names=None):
        """
        Create visualizations comparing original and synthetic data.

        Args:
            X_train: Original training features
            y_train: Original training labels
            X_resampled: Resampled features
            y_resampled: Resampled labels
            output_dir (str): Directory to save visualizations
            feature_names (list): Names of features
        """
        try:
            logger.info("Creating comparison visualizations...")
            os.makedirs(output_dir, exist_ok=True)

            # 1. Class Distribution Comparison
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Before SMOTE
            classes, counts = np.unique(y_train, return_counts=True)
            axes[0].bar(classes, counts, color=['green', 'red'], alpha=0.7, edgecolor='black')
            axes[0].set_title('Class Distribution - Before SMOTE', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Class', fontsize=12)
            axes[0].set_ylabel('Count', fontsize=12)
            axes[0].set_xticks(classes)
            axes[0].set_xticklabels(['No Diabetes', 'Diabetes'])
            for i, (c, count) in enumerate(zip(classes, counts)):
                axes[0].text(c, count + 5, str(count), ha='center', fontweight='bold')

            # After SMOTE
            classes, counts = np.unique(y_resampled, return_counts=True)
            axes[1].bar(classes, counts, color=['green', 'red'], alpha=0.7, edgecolor='black')
            axes[1].set_title('Class Distribution - After SMOTE', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Class', fontsize=12)
            axes[1].set_ylabel('Count', fontsize=12)
            axes[1].set_xticks(classes)
            axes[1].set_xticklabels(['No Diabetes', 'Diabetes'])
            for i, (c, count) in enumerate(zip(classes, counts)):
                axes[1].text(c, count + 5, str(count), ha='center', fontweight='bold')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'class_balance_comparison.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved: class_balance_comparison.png")

            # 2. PCA Visualization
            logger.info("Creating PCA visualization...")

            # Convert to numpy if needed
            if isinstance(X_train, pd.DataFrame):
                X_train_arr = X_train.values
                X_resampled_arr = X_resampled.values
            else:
                X_train_arr = X_train
                X_resampled_arr = X_resampled

            # Apply PCA
            pca = PCA(n_components=2, random_state=self.random_state)

            # Fit on original data
            X_train_pca = pca.fit_transform(X_train_arr)

            # Transform resampled data
            X_resampled_pca = pca.transform(X_resampled_arr)

            # Identify synthetic samples
            n_original = len(X_train)
            is_synthetic = np.zeros(len(X_resampled), dtype=bool)
            is_synthetic[n_original:] = True

            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Original data
            for class_val in np.unique(y_train):
                mask = y_train == class_val
                axes[0].scatter(X_train_pca[mask, 0], X_train_pca[mask, 1],
                              label=f'Class {class_val}', alpha=0.6, s=50)
            axes[0].set_title('Original Data (PCA)', fontsize=14, fontweight='bold')
            axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
            axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            # Resampled data with synthetic samples highlighted
            minority_class = min(self.original_distribution.keys(),
                               key=lambda x: self.original_distribution[x])

            # Plot original samples
            for class_val in np.unique(y_resampled):
                mask = (y_resampled == class_val) & (~is_synthetic)
                axes[1].scatter(X_resampled_pca[mask, 0], X_resampled_pca[mask, 1],
                              label=f'Class {class_val} (Original)', alpha=0.6, s=50)

            # Plot synthetic samples
            mask_synthetic = (y_resampled == minority_class) & is_synthetic
            axes[1].scatter(X_resampled_pca[mask_synthetic, 0],
                          X_resampled_pca[mask_synthetic, 1],
                          label=f'Class {minority_class} (Synthetic)',
                          alpha=0.8, s=50, marker='^', edgecolors='black', linewidth=0.5)

            axes[1].set_title('After SMOTE - Original vs Synthetic (PCA)', fontsize=14, fontweight='bold')
            axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
            axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'pca_comparison.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved: pca_comparison.png")

            # 3. Feature Distribution Comparison (for key features)
            if feature_names is None:
                if isinstance(X_train, pd.DataFrame):
                    feature_names = X_train.columns.tolist()
                else:
                    feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]

            # Select up to 6 key features
            n_features_to_plot = min(6, len(feature_names))
            key_features_idx = np.random.choice(len(feature_names), n_features_to_plot, replace=False)

            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.ravel()

            minority_class = min(self.original_distribution.keys(),
                               key=lambda x: self.original_distribution[x])

            for i, feat_idx in enumerate(key_features_idx):
                # Original minority class
                orig_minority_data = X_train_arr[y_train == minority_class, feat_idx]

                # Resampled minority class
                resamp_minority_data = X_resampled_arr[y_resampled == minority_class, feat_idx]

                # Plot histograms
                axes[i].hist(orig_minority_data, bins=30, alpha=0.5, label='Original',
                           color='blue', edgecolor='black')
                axes[i].hist(resamp_minority_data, bins=30, alpha=0.5, label='With Synthetic',
                           color='red', edgecolor='black')
                axes[i].set_title(f'{feature_names[feat_idx]} Distribution\n(Minority Class)',
                                fontsize=11, fontweight='bold')
                axes[i].set_xlabel(feature_names[feat_idx], fontsize=10)
                axes[i].set_ylabel('Frequency', fontsize=10)
                axes[i].legend()
                axes[i].grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'feature_distributions_comparison.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Saved: feature_distributions_comparison.png")

            logger.info("All visualizations created successfully!")

        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise

    def save_synthetic_data(self, X_resampled, y_resampled, output_dir='../data/synthetic'):
        """
        Save synthetic data to disk.

        Args:
            X_resampled: Resampled features
            y_resampled: Resampled labels
            output_dir (str): Directory to save the files
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving synthetic data to {output_dir}...")

            # Convert to DataFrame if needed
            if isinstance(X_resampled, np.ndarray):
                X_resampled = pd.DataFrame(X_resampled)

            if isinstance(y_resampled, np.ndarray):
                y_resampled = pd.Series(y_resampled, name='Outcome')

            # Save datasets
            X_resampled.to_csv(os.path.join(output_dir, 'X_train_resampled.csv'), index=False)
            y_resampled.to_csv(os.path.join(output_dir, 'y_train_resampled.csv'), index=False)

            logger.info("Synthetic data saved successfully!")
            logger.info(f"Files: X_train_resampled.csv, y_train_resampled.csv")

        except Exception as e:
            logger.error(f"Error saving synthetic data: {str(e)}")
            raise


def generate_synthetic_data(X_train, y_train, method='smote', sampling_strategy='auto',
                           k_neighbors=5, validate=True, visualize=True,
                           output_dir='../data/synthetic'):
    """
    Complete pipeline for generating and validating synthetic data.

    Args:
        X_train: Training features
        y_train: Training labels
        method (str): Sampling method ('smote', 'borderline', 'adasyn', 'svmsmote')
        sampling_strategy: Sampling strategy
        k_neighbors (int): Number of neighbors for SMOTE
        validate (bool): Whether to validate synthetic data
        visualize (bool): Whether to create visualizations
        output_dir (str): Directory to save outputs

    Returns:
        tuple: X_resampled, y_resampled, generator
    """
    try:
        # Initialize generator
        generator = SyntheticDataGenerator(random_state=42)

        # Generate synthetic data
        X_resampled, y_resampled = generator.generate_synthetic_data(
            X_train, y_train,
            method=method,
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors
        )

        # Validate synthetic data
        if validate:
            feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None
            validation_results = generator.validate_synthetic_data(
                X_train, y_train,
                X_resampled, y_resampled,
                feature_names=feature_names
            )

        # Create visualizations
        if visualize:
            feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None
            generator.visualize_comparison(
                X_train, y_train,
                X_resampled, y_resampled,
                output_dir=os.path.join(output_dir, '../results/eda'),
                feature_names=feature_names
            )

        # Save synthetic data
        generator.save_synthetic_data(X_resampled, y_resampled, output_dir)

        return X_resampled, y_resampled, generator

    except Exception as e:
        logger.error(f"Error in synthetic data generation pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    print("Synthetic Data Generation Module")
    print("="*80)
    print("\nThis module handles class imbalance using SMOTE and related techniques.")
    print("\nUsage example:")
    print("""
    from synthetic_data_generation import generate_synthetic_data
    import pandas as pd

    # Load preprocessed data
    X_train = pd.read_csv('../data/processed/X_train.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()

    # Generate synthetic data
    X_resampled, y_resampled, generator = generate_synthetic_data(
        X_train, y_train,
        method='smote',
        sampling_strategy='auto',
        k_neighbors=5,
        validate=True,
        visualize=True
    )
    """)
