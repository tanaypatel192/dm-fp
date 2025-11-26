"""
Random Forest Model for Diabetes Prediction

This module implements a comprehensive Random Forest classifier with:
- Hyperparameter tuning using GridSearchCV
- Detailed evaluation metrics
- Feature importance analysis with partial dependence plots
- Out-of-bag (OOB) error analysis
- Tree diversity analysis
- Comparison with single decision tree

Author: Diabetes Prediction Project
Date: 2025
"""

import pandas as pd
import numpy as np
import logging
import os
import joblib
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RandomForestModel:
    """
    A comprehensive Random Forest classifier for diabetes prediction.

    This class provides methods for training, evaluation, feature importance,
    partial dependence analysis, OOB error analysis, and tree diversity metrics.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the RandomForestModel.

        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.model: Optional[RandomForestClassifier] = None
        self.best_params: Optional[Dict] = None
        self.feature_names: Optional[List[str]] = None
        self.class_names: List[str] = ['No Diabetes', 'Diabetes']
        self.cv_results: Optional[Dict] = None
        self.grid_search: Optional[GridSearchCV] = None
        self.oob_scores: Optional[List[float]] = None

        logger.info("RandomForestModel initialized")

    def train(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        param_grid: Optional[Dict] = None,
        cv: int = 10,
        scoring: str = 'f1',
        n_jobs: int = -1
    ) -> 'RandomForestModel':
        """
        Train the Random Forest model with hyperparameter tuning using GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Dictionary of parameters to tune. If None, uses default grid.
            cv: Number of cross-validation folds (default: 10)
            scoring: Scoring metric for GridSearchCV (default: 'f1')
            n_jobs: Number of parallel jobs (default: -1 for all processors)

        Returns:
            self: Returns self for method chaining
        """
        try:
            logger.info("="*80)
            logger.info("TRAINING RANDOM FOREST MODEL")
            logger.info("="*80)

            # Extract feature names
            if isinstance(X_train, pd.DataFrame):
                self.feature_names = X_train.columns.tolist()
                X_train_arr = X_train.values
            else:
                self.feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
                X_train_arr = X_train

            if isinstance(y_train, pd.Series):
                y_train_arr = y_train.values
            else:
                y_train_arr = y_train

            logger.info(f"Training data shape: {X_train_arr.shape}")
            logger.info(f"Number of features: {len(self.feature_names)}")
            logger.info(f"Class distribution: {np.bincount(y_train_arr)}")

            # Default parameter grid if not provided
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4, 5],
                    'max_features': ['sqrt', 'log2', None],
                    'bootstrap': [True]  # Enable for OOB scoring
                }

            logger.info(f"Hyperparameter grid:")
            for param, values in param_grid.items():
                logger.info(f"  {param}: {values}")

            logger.info(f"Cross-validation folds: {cv}")
            logger.info(f"Scoring metric: {scoring}")
            logger.info(f"Parallel processing: Using all available cores (n_jobs={n_jobs})")

            # Initialize base estimator with OOB score enabled
            base_estimator = RandomForestClassifier(
                random_state=self.random_state,
                oob_score=True,
                n_jobs=n_jobs
            )

            # Perform grid search
            logger.info("Starting GridSearchCV...")
            logger.info("This may take several minutes due to the large parameter space...")

            self.grid_search = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1,
                return_train_score=True
            )

            # Fit the model
            self.grid_search.fit(X_train_arr, y_train_arr)

            # Store best model and parameters
            self.model = self.grid_search.best_estimator_
            self.best_params = self.grid_search.best_params_

            logger.info("="*80)
            logger.info("TRAINING COMPLETED")
            logger.info("="*80)
            logger.info(f"Best parameters found:")
            for param, value in self.best_params.items():
                logger.info(f"  {param}: {value}")
            logger.info(f"Best cross-validation {scoring} score: {self.grid_search.best_score_:.4f}")

            # Log OOB score
            if hasattr(self.model, 'oob_score_'):
                logger.info(f"Out-of-bag score: {self.model.oob_score_:.4f}")

            # Store CV results
            self.cv_results = {
                'best_score': self.grid_search.best_score_,
                'best_params': self.best_params,
                'cv_results': pd.DataFrame(self.grid_search.cv_results_)
            }

            # Perform additional cross-validation with best model
            logger.info(f"\nPerforming {cv}-fold cross-validation with best model...")
            cv_scores = cross_val_score(
                self.model, X_train_arr, y_train_arr,
                cv=cv, scoring=scoring, n_jobs=n_jobs
            )

            logger.info(f"Cross-validation {scoring} scores: {cv_scores}")
            logger.info(f"Mean CV {scoring}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

            # Store CV scores
            self.cv_results['cv_scores'] = cv_scores
            self.cv_results['mean_cv_score'] = cv_scores.mean()
            self.cv_results['std_cv_score'] = cv_scores.std()

            return self

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def evaluate(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        output_dir: str = '../results/random_forest'
    ) -> Dict[str, Any]:
        """
        Evaluate the model and calculate comprehensive metrics.

        Args:
            X_test: Test features
            y_test: Test labels
            output_dir: Directory to save evaluation results

        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet. Call train() first.")

            logger.info("="*80)
            logger.info("EVALUATING RANDOM FOREST MODEL")
            logger.info("="*80)

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Convert to numpy arrays if needed
            if isinstance(X_test, pd.DataFrame):
                X_test_arr = X_test.values
            else:
                X_test_arr = X_test

            if isinstance(y_test, pd.Series):
                y_test_arr = y_test.values
            else:
                y_test_arr = y_test

            # Make predictions
            y_pred = self.model.predict(X_test_arr)
            y_pred_proba = self.model.predict_proba(X_test_arr)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test_arr, y_pred)
            precision = precision_score(y_test_arr, y_pred, zero_division=0)
            recall = recall_score(y_test_arr, y_pred, zero_division=0)
            f1 = f1_score(y_test_arr, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test_arr, y_pred_proba)

            # Confusion matrix
            cm = confusion_matrix(y_test_arr, y_pred)

            # Classification report
            class_report = classification_report(
                y_test_arr, y_pred,
                target_names=self.class_names,
                output_dict=True
            )

            # Log metrics
            logger.info(f"\nTest Set Performance:")
            logger.info(f"  Accuracy:  {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall:    {recall:.4f}")
            logger.info(f"  F1-Score:  {f1:.4f}")
            logger.info(f"  ROC-AUC:   {roc_auc:.4f}")

            logger.info(f"\nConfusion Matrix:")
            logger.info(f"\n{cm}")

            logger.info(f"\nClassification Report:")
            logger.info(f"\n{classification_report(y_test_arr, y_pred, target_names=self.class_names)}")

            # Create evaluation results dictionary
            evaluation_results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': cm,
                'classification_report': class_report,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'y_test': y_test_arr
            }

            # Visualize confusion matrix
            self._plot_confusion_matrix(cm, output_dir)

            # Plot ROC curve
            self._plot_roc_curve(y_test_arr, y_pred_proba, roc_auc, output_dir)

            # Plot Precision-Recall curve
            self._plot_precision_recall_curve(y_test_arr, y_pred_proba, output_dir)

            # Save metrics to file
            self._save_metrics(evaluation_results, output_dir)

            logger.info(f"\nEvaluation results saved to: {output_dir}")
            logger.info("="*80)

            return evaluation_results

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def get_feature_importance(
        self,
        output_dir: str = '../results/random_forest',
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Extract and visualize feature importance from Random Forest.

        Args:
            output_dir: Directory to save visualizations
            top_n: Number of top features to display

        Returns:
            DataFrame: Feature importance scores
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet. Call train() first.")

            logger.info("Extracting feature importance from Random Forest...")
            os.makedirs(output_dir, exist_ok=True)

            # Get feature importance (mean decrease in impurity)
            importance = self.model.feature_importances_

            # Calculate standard deviation across trees
            importances_std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)

            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance,
                'std': importances_std
            }).sort_values('importance', ascending=False)

            # Log top features
            logger.info(f"\nTop {top_n} Most Important Features:")
            logger.info("="*70)
            for idx, row in importance_df.head(top_n).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f} (+/- {row['std']:.4f})")

            # Visualize top N features with error bars
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(top_n)

            # Create positions for bars
            positions = np.arange(len(top_features))
            colors = plt.cm.RdYlGn(top_features['importance'] / top_features['importance'].max())

            # Plot with error bars
            bars = plt.barh(positions, top_features['importance'],
                          xerr=top_features['std'], color=colors,
                          edgecolor='black', linewidth=1.5, capsize=5)

            plt.yticks(positions, top_features['feature'], fontsize=11)
            plt.xlabel('Importance Score (Mean Decrease in Impurity)', fontsize=13, fontweight='bold')
            plt.ylabel('Features', fontsize=13, fontweight='bold')
            plt.title(f'Top {top_n} Most Important Features - Random Forest',
                     fontsize=15, fontweight='bold')
            plt.gca().invert_yaxis()

            # Add value labels
            for i, (idx, row) in enumerate(top_features.iterrows()):
                plt.text(row['importance'], i, f" {row['importance']:.4f}",
                        va='center', fontweight='bold')

            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()

            top_n_path = os.path.join(output_dir, f'feature_importance_top_{top_n}.png')
            plt.savefig(top_n_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Feature importance plot saved to: {top_n_path}")

            # Create feature importance ranking table
            self._create_importance_table(importance_df, output_dir)

            # Save to CSV
            csv_path = os.path.join(output_dir, 'feature_importance.csv')
            importance_df.to_csv(csv_path, index=False)
            logger.info(f"Feature importance data saved to: {csv_path}")

            return importance_df

        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
            raise

    def plot_partial_dependence(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        output_dir: str = '../results/random_forest',
        top_n: int = 3
    ) -> None:
        """
        Create partial dependence plots for top N features.

        Args:
            X: Training or test features
            output_dir: Directory to save plots
            top_n: Number of top features to plot
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet. Call train() first.")

            logger.info(f"Creating partial dependence plots for top {top_n} features...")
            os.makedirs(output_dir, exist_ok=True)

            # Convert to numpy array if needed
            if isinstance(X, pd.DataFrame):
                X_arr = X.values
            else:
                X_arr = X

            # Get top N important features
            importance = self.model.feature_importances_
            top_indices = np.argsort(importance)[-top_n:][::-1]
            top_features = [self.feature_names[i] for i in top_indices]

            logger.info(f"Creating partial dependence plots for: {top_features}")

            # Create individual plots for each feature
            for feature_idx, feature_name in zip(top_indices, top_features):
                fig, ax = plt.subplots(figsize=(10, 6))

                # Calculate partial dependence
                pd_result = partial_dependence(
                    self.model, X_arr, [feature_idx],
                    kind='average', grid_resolution=50
                )

                # Plot
                ax.plot(pd_result['values'][0], pd_result['average'][0],
                       linewidth=2, color='darkblue')
                ax.set_xlabel(feature_name, fontsize=12, fontweight='bold')
                ax.set_ylabel('Partial Dependence', fontsize=12, fontweight='bold')
                ax.set_title(f'Partial Dependence Plot - {feature_name}',
                           fontsize=14, fontweight='bold')
                ax.grid(alpha=0.3)

                plt.tight_layout()
                plot_path = os.path.join(output_dir, f'partial_dependence_{feature_name}.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f"Saved partial dependence plot for {feature_name}")

            # Create combined plot
            fig, axes = plt.subplots(1, top_n, figsize=(6*top_n, 5))
            if top_n == 1:
                axes = [axes]

            display = PartialDependenceDisplay.from_estimator(
                self.model, X_arr, top_indices,
                feature_names=self.feature_names,
                kind='average', grid_resolution=50,
                ax=axes
            )

            fig.suptitle('Partial Dependence Plots - Top Features',
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()

            combined_path = os.path.join(output_dir, 'partial_dependence_combined.png')
            plt.savefig(combined_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Combined partial dependence plot saved to: {combined_path}")

        except Exception as e:
            logger.error(f"Error creating partial dependence plots: {str(e)}")
            raise

    def analyze_oob_error(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        output_dir: str = '../results/random_forest',
        max_estimators: int = 300
    ) -> Dict[str, Any]:
        """
        Calculate and plot out-of-bag (OOB) error as a function of number of trees.

        Args:
            X_train: Training features
            y_train: Training labels
            output_dir: Directory to save plots
            max_estimators: Maximum number of estimators to test

        Returns:
            dict: OOB analysis results
        """
        try:
            logger.info("Analyzing out-of-bag (OOB) error...")
            os.makedirs(output_dir, exist_ok=True)

            # Convert to numpy arrays if needed
            if isinstance(X_train, pd.DataFrame):
                X_train_arr = X_train.values
            else:
                X_train_arr = X_train

            if isinstance(y_train, pd.Series):
                y_train_arr = y_train.values
            else:
                y_train_arr = y_train

            # Test different numbers of estimators
            estimator_range = range(10, max_estimators + 1, 10)
            oob_scores = []
            oob_errors = []

            logger.info(f"Testing estimators from 10 to {max_estimators} (step=10)...")

            for n_est in estimator_range:
                # Train RF with current number of estimators
                rf = RandomForestClassifier(
                    n_estimators=n_est,
                    oob_score=True,
                    random_state=self.random_state,
                    max_depth=self.best_params.get('max_depth') if self.best_params else None,
                    min_samples_split=self.best_params.get('min_samples_split', 2) if self.best_params else 2,
                    min_samples_leaf=self.best_params.get('min_samples_leaf', 1) if self.best_params else 1,
                    max_features=self.best_params.get('max_features', 'sqrt') if self.best_params else 'sqrt',
                    n_jobs=-1
                )

                rf.fit(X_train_arr, y_train_arr)
                oob_score = rf.oob_score_
                oob_error = 1 - oob_score

                oob_scores.append(oob_score)
                oob_errors.append(oob_error)

            self.oob_scores = oob_scores

            # Plot OOB error
            plt.figure(figsize=(12, 6))
            plt.plot(list(estimator_range), oob_errors, marker='o',
                    linewidth=2, markersize=4, color='darkred', label='OOB Error')
            plt.plot(list(estimator_range), [1-s for s in oob_scores], marker='s',
                    linewidth=2, markersize=4, color='darkgreen', alpha=0.5, label='Test Error Rate')
            plt.xlabel('Number of Trees', fontsize=13, fontweight='bold')
            plt.ylabel('Error Rate', fontsize=13, fontweight='bold')
            plt.title('Out-of-Bag Error vs Number of Trees', fontsize=15, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(alpha=0.3)
            plt.tight_layout()

            oob_path = os.path.join(output_dir, 'oob_error_analysis.png')
            plt.savefig(oob_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"OOB error plot saved to: {oob_path}")

            # Find optimal number of trees
            optimal_idx = np.argmin(oob_errors)
            optimal_n_trees = list(estimator_range)[optimal_idx]
            optimal_oob_error = oob_errors[optimal_idx]

            logger.info(f"Optimal number of trees: {optimal_n_trees}")
            logger.info(f"Minimum OOB error: {optimal_oob_error:.4f}")
            logger.info(f"Maximum OOB accuracy: {1-optimal_oob_error:.4f}")

            # Create results dictionary
            oob_results = {
                'estimator_range': list(estimator_range),
                'oob_scores': oob_scores,
                'oob_errors': oob_errors,
                'optimal_n_trees': optimal_n_trees,
                'optimal_oob_error': optimal_oob_error,
                'optimal_oob_score': 1 - optimal_oob_error
            }

            # Save to file
            oob_file = os.path.join(output_dir, 'oob_analysis.txt')
            with open(oob_file, 'w') as f:
                f.write("OUT-OF-BAG (OOB) ERROR ANALYSIS\n")
                f.write("="*80 + "\n\n")
                f.write(f"Optimal number of trees: {optimal_n_trees}\n")
                f.write(f"Minimum OOB error: {optimal_oob_error:.4f}\n")
                f.write(f"Maximum OOB accuracy: {1-optimal_oob_error:.4f}\n\n")
                f.write("OOB Error by Number of Trees:\n")
                f.write("-"*40 + "\n")
                for n_est, score, error in zip(estimator_range, oob_scores, oob_errors):
                    f.write(f"{n_est:3d} trees: OOB Score = {score:.4f}, OOB Error = {error:.4f}\n")

            logger.info(f"OOB analysis saved to: {oob_file}")

            return oob_results

        except Exception as e:
            logger.error(f"Error analyzing OOB error: {str(e)}")
            raise

    def get_tree_diversity(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        output_dir: str = '../results/random_forest'
    ) -> Dict[str, Any]:
        """
        Analyze diversity of trees in the Random Forest.

        Args:
            X: Features to use for prediction diversity analysis
            output_dir: Directory to save results

        Returns:
            dict: Tree diversity metrics
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet. Call train() first.")

            logger.info("Analyzing tree diversity in Random Forest...")
            os.makedirs(output_dir, exist_ok=True)

            # Convert to numpy array if needed
            if isinstance(X, pd.DataFrame):
                X_arr = X.values
            else:
                X_arr = X

            # Get predictions from all trees
            all_predictions = np.array([tree.predict(X_arr) for tree in self.model.estimators_])

            # Calculate pairwise agreement between trees
            n_trees = len(self.model.estimators_)
            agreements = []

            logger.info(f"Calculating pairwise agreement between {n_trees} trees...")

            for i in range(n_trees):
                for j in range(i + 1, n_trees):
                    agreement = np.mean(all_predictions[i] == all_predictions[j])
                    agreements.append(agreement)

            mean_agreement = np.mean(agreements)
            std_agreement = np.std(agreements)

            # Calculate tree depths
            tree_depths = [tree.tree_.max_depth for tree in self.model.estimators_]
            mean_depth = np.mean(tree_depths)
            std_depth = np.std(tree_depths)

            # Calculate number of leaves
            n_leaves = [tree.tree_.n_leaves for tree in self.model.estimators_]
            mean_leaves = np.mean(n_leaves)
            std_leaves = np.std(n_leaves)

            # Calculate number of features used by each tree
            features_used = []
            for tree in self.model.estimators_:
                features_in_tree = np.unique(tree.tree_.feature[tree.tree_.feature >= 0])
                features_used.append(len(features_in_tree))

            mean_features = np.mean(features_used)
            std_features = np.std(features_used)

            logger.info(f"\nTree Diversity Metrics:")
            logger.info(f"  Mean pairwise agreement: {mean_agreement:.4f} (+/- {std_agreement:.4f})")
            logger.info(f"  Mean tree depth: {mean_depth:.2f} (+/- {std_depth:.2f})")
            logger.info(f"  Mean number of leaves: {mean_leaves:.2f} (+/- {std_leaves:.2f})")
            logger.info(f"  Mean features used: {mean_features:.2f} (+/- {std_features:.2f})")

            # Visualize diversity metrics
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Plot 1: Tree depths distribution
            axes[0, 0].hist(tree_depths, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
            axes[0, 0].axvline(mean_depth, color='red', linestyle='--', linewidth=2,
                              label=f'Mean: {mean_depth:.2f}')
            axes[0, 0].set_xlabel('Tree Depth', fontsize=11, fontweight='bold')
            axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
            axes[0, 0].set_title('Distribution of Tree Depths', fontsize=12, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)

            # Plot 2: Number of leaves distribution
            axes[0, 1].hist(n_leaves, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
            axes[0, 1].axvline(mean_leaves, color='red', linestyle='--', linewidth=2,
                              label=f'Mean: {mean_leaves:.2f}')
            axes[0, 1].set_xlabel('Number of Leaves', fontsize=11, fontweight='bold')
            axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
            axes[0, 1].set_title('Distribution of Number of Leaves', fontsize=12, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)

            # Plot 3: Features used distribution
            axes[1, 0].hist(features_used, bins=20, edgecolor='black', alpha=0.7, color='salmon')
            axes[1, 0].axvline(mean_features, color='red', linestyle='--', linewidth=2,
                              label=f'Mean: {mean_features:.2f}')
            axes[1, 0].set_xlabel('Number of Features Used', fontsize=11, fontweight='bold')
            axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
            axes[1, 0].set_title('Distribution of Features Used per Tree', fontsize=12, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)

            # Plot 4: Pairwise agreement distribution
            axes[1, 1].hist(agreements, bins=30, edgecolor='black', alpha=0.7, color='plum')
            axes[1, 1].axvline(mean_agreement, color='red', linestyle='--', linewidth=2,
                              label=f'Mean: {mean_agreement:.4f}')
            axes[1, 1].set_xlabel('Pairwise Agreement', fontsize=11, fontweight='bold')
            axes[1, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
            axes[1, 1].set_title('Distribution of Pairwise Tree Agreement', fontsize=12, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)

            plt.tight_layout()
            diversity_path = os.path.join(output_dir, 'tree_diversity_analysis.png')
            plt.savefig(diversity_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Tree diversity visualization saved to: {diversity_path}")

            # Create results dictionary
            diversity_results = {
                'mean_agreement': mean_agreement,
                'std_agreement': std_agreement,
                'mean_depth': mean_depth,
                'std_depth': std_depth,
                'mean_leaves': mean_leaves,
                'std_leaves': std_leaves,
                'mean_features_used': mean_features,
                'std_features_used': std_features,
                'tree_depths': tree_depths,
                'n_leaves': n_leaves,
                'features_used': features_used,
                'pairwise_agreements': agreements
            }

            # Save to file
            diversity_file = os.path.join(output_dir, 'tree_diversity_metrics.txt')
            with open(diversity_file, 'w') as f:
                f.write("TREE DIVERSITY ANALYSIS\n")
                f.write("="*80 + "\n\n")
                f.write(f"Number of trees in forest: {n_trees}\n\n")
                f.write("Diversity Metrics:\n")
                f.write("-"*40 + "\n")
                f.write(f"Mean pairwise agreement: {mean_agreement:.4f} (+/- {std_agreement:.4f})\n")
                f.write(f"Mean tree depth: {mean_depth:.2f} (+/- {std_depth:.2f})\n")
                f.write(f"Mean number of leaves: {mean_leaves:.2f} (+/- {std_leaves:.2f})\n")
                f.write(f"Mean features used per tree: {mean_features:.2f} (+/- {std_features:.2f})\n")
                f.write(f"Total features available: {len(self.feature_names)}\n\n")
                f.write("Interpretation:\n")
                f.write("-"*40 + "\n")
                f.write("- Lower pairwise agreement indicates higher diversity (better)\n")
                f.write("- Higher diversity generally leads to better generalization\n")
                f.write("- Trees should use different subsets of features\n")

            logger.info(f"Tree diversity metrics saved to: {diversity_file}")

            return diversity_results

        except Exception as e:
            logger.error(f"Error analyzing tree diversity: {str(e)}")
            raise

    def compare_with_decision_tree(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        output_dir: str = '../results/random_forest'
    ) -> Dict[str, Any]:
        """
        Compare Random Forest performance with a single Decision Tree.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            output_dir: Directory to save comparison results

        Returns:
            dict: Comparison results
        """
        try:
            if self.model is None:
                raise ValueError("Random Forest model has not been trained yet. Call train() first.")

            logger.info("="*80)
            logger.info("COMPARING RANDOM FOREST WITH SINGLE DECISION TREE")
            logger.info("="*80)

            os.makedirs(output_dir, exist_ok=True)

            # Convert to numpy arrays if needed
            if isinstance(X_train, pd.DataFrame):
                X_train_arr = X_train.values
            else:
                X_train_arr = X_train

            if isinstance(X_test, pd.DataFrame):
                X_test_arr = X_test.values
            else:
                X_test_arr = X_test

            if isinstance(y_train, pd.Series):
                y_train_arr = y_train.values
            else:
                y_train_arr = y_train

            if isinstance(y_test, pd.Series):
                y_test_arr = y_test.values
            else:
                y_test_arr = y_test

            # Train a single decision tree with similar parameters
            logger.info("Training single Decision Tree for comparison...")
            dt_model = DecisionTreeClassifier(
                max_depth=self.best_params.get('max_depth'),
                min_samples_split=self.best_params.get('min_samples_split', 2),
                min_samples_leaf=self.best_params.get('min_samples_leaf', 1),
                random_state=self.random_state
            )
            dt_model.fit(X_train_arr, y_train_arr)

            # Get predictions for both models
            rf_pred = self.model.predict(X_test_arr)
            rf_pred_proba = self.model.predict_proba(X_test_arr)[:, 1]

            dt_pred = dt_model.predict(X_test_arr)
            dt_pred_proba = dt_model.predict_proba(X_test_arr)[:, 1]

            # Calculate metrics for both
            rf_metrics = {
                'accuracy': accuracy_score(y_test_arr, rf_pred),
                'precision': precision_score(y_test_arr, rf_pred, zero_division=0),
                'recall': recall_score(y_test_arr, rf_pred, zero_division=0),
                'f1': f1_score(y_test_arr, rf_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test_arr, rf_pred_proba)
            }

            dt_metrics = {
                'accuracy': accuracy_score(y_test_arr, dt_pred),
                'precision': precision_score(y_test_arr, dt_pred, zero_division=0),
                'recall': recall_score(y_test_arr, dt_pred, zero_division=0),
                'f1': f1_score(y_test_arr, dt_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test_arr, dt_pred_proba)
            }

            # Log comparison
            logger.info("\nPerformance Comparison:")
            logger.info("="*70)
            logger.info(f"{'Metric':<15} {'Random Forest':<20} {'Decision Tree':<20} {'Improvement':<15}")
            logger.info("-"*70)

            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                rf_val = rf_metrics[metric]
                dt_val = dt_metrics[metric]
                improvement = ((rf_val - dt_val) / dt_val) * 100 if dt_val > 0 else 0
                logger.info(f"{metric.upper():<15} {rf_val:<20.4f} {dt_val:<20.4f} {improvement:+.2f}%")

            # Visualize comparison
            metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            rf_values = [rf_metrics[m] for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
            dt_values = [dt_metrics[m] for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]

            x = np.arange(len(metrics_names))
            width = 0.35

            fig, ax = plt.subplots(figsize=(12, 6))
            bars1 = ax.bar(x - width/2, rf_values, width, label='Random Forest',
                          color='darkgreen', edgecolor='black', linewidth=1.5)
            bars2 = ax.bar(x + width/2, dt_values, width, label='Decision Tree',
                          color='steelblue', edgecolor='black', linewidth=1.5)

            ax.set_xlabel('Metrics', fontsize=13, fontweight='bold')
            ax.set_ylabel('Score', fontsize=13, fontweight='bold')
            ax.set_title('Random Forest vs Decision Tree - Performance Comparison',
                        fontsize=15, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_names)
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1.1])

            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            comparison_path = os.path.join(output_dir, 'rf_vs_dt_comparison.png')
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"\nComparison visualization saved to: {comparison_path}")

            # Save comparison to file
            comparison_file = os.path.join(output_dir, 'rf_vs_dt_comparison.txt')
            with open(comparison_file, 'w') as f:
                f.write("RANDOM FOREST VS DECISION TREE COMPARISON\n")
                f.write("="*80 + "\n\n")
                f.write(f"{'Metric':<15} {'Random Forest':<20} {'Decision Tree':<20} {'Improvement':<15}\n")
                f.write("-"*80 + "\n")

                for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                    rf_val = rf_metrics[metric]
                    dt_val = dt_metrics[metric]
                    improvement = ((rf_val - dt_val) / dt_val) * 100 if dt_val > 0 else 0
                    f.write(f"{metric.upper():<15} {rf_val:<20.4f} {dt_val:<20.4f} {improvement:+.2f}%\n")

            logger.info(f"Comparison results saved to: {comparison_file}")

            comparison_results = {
                'random_forest_metrics': rf_metrics,
                'decision_tree_metrics': dt_metrics,
                'improvements': {
                    metric: ((rf_metrics[metric] - dt_metrics[metric]) / dt_metrics[metric]) * 100
                    if dt_metrics[metric] > 0 else 0
                    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                }
            }

            logger.info("="*80)
            return comparison_results

        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            raise

    def save_model(self, output_dir: str = '../models', model_name: str = 'random_forest_model.pkl') -> None:
        """
        Save the trained model to disk.

        Args:
            output_dir: Directory to save the model
            model_name: Name of the model file
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet. Call train() first.")

            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, model_name)

            # Save model and metadata
            model_data = {
                'model': self.model,
                'best_params': self.best_params,
                'feature_names': self.feature_names,
                'class_names': self.class_names,
                'cv_results': self.cv_results,
                'oob_scores': self.oob_scores
            }

            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to: {model_path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path: str) -> 'RandomForestModel':
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the saved model file

        Returns:
            self: Returns self for method chaining
        """
        try:
            logger.info(f"Loading model from: {model_path}")
            model_data = joblib.load(model_path)

            self.model = model_data['model']
            self.best_params = model_data['best_params']
            self.feature_names = model_data['feature_names']
            self.class_names = model_data['class_names']
            self.cv_results = model_data.get('cv_results')
            self.oob_scores = model_data.get('oob_scores')

            logger.info("Model loaded successfully")
            return self

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _plot_confusion_matrix(self, cm: np.ndarray, output_dir: str) -> None:
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'},
            linewidths=2, linecolor='black'
        )
        plt.title('Confusion Matrix - Random Forest', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=13, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
        plt.tight_layout()

        output_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Confusion matrix saved to: {output_path}")

    def _plot_roc_curve(self, y_test: np.ndarray, y_pred_proba: np.ndarray,
                        roc_auc: float, output_dir: str) -> None:
        """Plot and save ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        plt.title('ROC Curve - Random Forest', fontsize=15, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"ROC curve saved to: {output_path}")

    def _plot_precision_recall_curve(self, y_test: np.ndarray,
                                     y_pred_proba: np.ndarray, output_dir: str) -> None:
        """Plot and save Precision-Recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkgreen', lw=2,
                label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall', fontsize=13, fontweight='bold')
        plt.ylabel('Precision', fontsize=13, fontweight='bold')
        plt.title('Precision-Recall Curve - Random Forest', fontsize=15, fontweight='bold')
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tight_layout()

        output_path = os.path.join(output_dir, 'precision_recall_curve.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Precision-Recall curve saved to: {output_path}")

    def _save_metrics(self, evaluation_results: Dict, output_dir: str) -> None:
        """Save evaluation metrics to file."""
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.txt')

        with open(metrics_path, 'w') as f:
            f.write("RANDOM FOREST MODEL - EVALUATION METRICS\n")
            f.write("="*80 + "\n\n")

            f.write("MODEL CONFIGURATION:\n")
            f.write("-"*40 + "\n")
            if self.best_params:
                for param, value in self.best_params.items():
                    f.write(f"{param}: {value}\n")
            f.write("\n")

            f.write("PERFORMANCE METRICS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Accuracy:  {evaluation_results['accuracy']:.4f}\n")
            f.write(f"Precision: {evaluation_results['precision']:.4f}\n")
            f.write(f"Recall:    {evaluation_results['recall']:.4f}\n")
            f.write(f"F1-Score:  {evaluation_results['f1_score']:.4f}\n")
            f.write(f"ROC-AUC:   {evaluation_results['roc_auc']:.4f}\n\n")

            if hasattr(self.model, 'oob_score_'):
                f.write(f"Out-of-Bag Score: {self.model.oob_score_:.4f}\n\n")

            f.write("CONFUSION MATRIX:\n")
            f.write("-"*40 + "\n")
            f.write(f"{evaluation_results['confusion_matrix']}\n\n")

            f.write("DETAILED CLASSIFICATION REPORT:\n")
            f.write("-"*40 + "\n")
            report_df = pd.DataFrame(evaluation_results['classification_report']).transpose()
            f.write(report_df.to_string())

        logger.info(f"Evaluation metrics saved to: {metrics_path}")

    def _create_importance_table(self, importance_df: pd.DataFrame, output_dir: str) -> None:
        """Create a formatted feature importance ranking table."""
        table_path = os.path.join(output_dir, 'feature_importance_table.txt')

        with open(table_path, 'w') as f:
            f.write("FEATURE IMPORTANCE RANKING - RANDOM FOREST\n")
            f.write("="*80 + "\n\n")
            f.write(f"{'Rank':<6} {'Feature':<30} {'Importance':<15} {'Std Dev':<15}\n")
            f.write("-"*80 + "\n")

            for rank, (idx, row) in enumerate(importance_df.iterrows(), 1):
                f.write(f"{rank:<6} {row['feature']:<30} {row['importance']:<15.6f} {row['std']:<15.6f}\n")

        logger.info(f"Feature importance table saved to: {table_path}")


def train_and_evaluate_random_forest(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    output_dir: str = '../results/random_forest',
    param_grid: Optional[Dict] = None,
    cv: int = 10
) -> RandomForestModel:
    """
    Complete pipeline for training and evaluating a Random Forest model.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        output_dir: Directory to save results
        param_grid: Parameter grid for GridSearchCV
        cv: Number of cross-validation folds

    Returns:
        RandomForestModel: Trained model
    """
    try:
        logger.info("="*80)
        logger.info("RANDOM FOREST MODEL - COMPLETE PIPELINE")
        logger.info("="*80)

        # Initialize model
        rf_model = RandomForestModel(random_state=42)

        # Train model
        rf_model.train(X_train, y_train, param_grid=param_grid, cv=cv, n_jobs=-1)

        # Evaluate model
        rf_model.evaluate(X_test, y_test, output_dir=output_dir)

        # Get feature importance
        rf_model.get_feature_importance(output_dir=output_dir, top_n=10)

        # Plot partial dependence for top 3 features
        rf_model.plot_partial_dependence(X_train, output_dir=output_dir, top_n=3)

        # Analyze OOB error
        rf_model.analyze_oob_error(X_train, y_train, output_dir=output_dir, max_estimators=300)

        # Analyze tree diversity
        rf_model.get_tree_diversity(X_test, output_dir=output_dir)

        # Compare with Decision Tree
        rf_model.compare_with_decision_tree(X_train, X_test, y_train, y_test, output_dir=output_dir)

        # Save model
        rf_model.save_model(output_dir='../models')

        logger.info("\n" + "="*80)
        logger.info("RANDOM FOREST PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)

        return rf_model

    except Exception as e:
        logger.error(f"Error in Random Forest pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    print("Random Forest Model for Diabetes Prediction")
    print("="*80)
    print("\nThis module provides a comprehensive Random Forest implementation.")
    print("\nUsage example:")
    print("""
    from random_forest_model import train_and_evaluate_random_forest
    import pandas as pd

    # Load preprocessed data
    X_train = pd.read_csv('../data/processed/X_train.csv')
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('../data/processed/y_test.csv').values.ravel()

    # Train and evaluate
    rf_model = train_and_evaluate_random_forest(
        X_train, X_test, y_train, y_test,
        output_dir='../results/random_forest'
    )
    """)
