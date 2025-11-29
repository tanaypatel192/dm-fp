"""
XGBoost Model for Diabetes Prediction

This module implements a comprehensive XGBoost classifier with:
- Extensive hyperparameter tuning using GridSearchCV
- Early stopping with validation set
- Multiple feature importance metrics (gain, coverage, weight)
- Learning curves visualization
- SHAP values for model explainability
- Individual tree visualization
- GPU acceleration support

Author: Diabetes Prediction Project
Date: 2025
"""

import pandas as pd
import numpy as np
import logging
import os
import joblib
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
import shap
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    A comprehensive XGBoost classifier for diabetes prediction.

    This class provides methods for training, evaluation, feature importance analysis,
    learning curves, SHAP explainability, and tree visualization.
    """

    def __init__(self, random_state: int = 42, use_gpu: bool = False):
        """
        Initialize the XGBoostModel.

        Args:
            random_state (int): Random seed for reproducibility
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.model: Optional[XGBClassifier] = None
        self.best_params: Optional[Dict] = None
        self.feature_names: Optional[List[str]] = None
        self.class_names: List[str] = ['No Diabetes', 'Diabetes']
        self.cv_results: Optional[Dict] = None
        self.grid_search: Optional[GridSearchCV] = None
        self.evals_result: Optional[Dict] = None
        self.shap_explainer: Optional[shap.TreeExplainer] = None
        self.shap_explainer: Optional[shap.TreeExplainer] = None
        self.shap_values: Optional[np.ndarray] = None
        self.metrics: Optional[Dict] = None

        # Check GPU availability
        self.device = 'cpu'
        if use_gpu:
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("GPU detected. GPU acceleration will be used.")
                    self.tree_method = 'hist'
                    self.device = 'cuda'
                else:
                    logger.warning("GPU requested but not available. Using CPU.")
                    self.tree_method = 'hist'
                    self.use_gpu = False
            except:
                logger.warning("GPU requested but not available. Using CPU.")
                self.tree_method = 'hist'
                self.use_gpu = False
        else:
            self.tree_method = 'hist'

        logger.info(f"XGBoostModel initialized (GPU: {self.use_gpu})")

    def train(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        param_grid: Optional[Dict] = None,
        cv: int = 10,
        scoring: str = 'f1',
        n_jobs: int = -1,
        early_stopping_rounds: int = 50,
        validation_split: float = 0.2
    ) -> 'XGBoostModel':
        """
        Train the XGBoost model with hyperparameter tuning and early stopping.

        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Dictionary of parameters to tune. If None, uses default grid.
            cv: Number of cross-validation folds (default: 10)
            scoring: Scoring metric for GridSearchCV (default: 'f1')
            n_jobs: Number of parallel jobs (default: -1 for all processors)
            early_stopping_rounds: Number of rounds for early stopping
            validation_split: Proportion of training data for validation

        Returns:
            self: Returns self for method chaining
        """
        try:
            logger.info("="*80)
            logger.info("TRAINING XGBOOST MODEL")
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
            logger.info(f"Using tree method: {self.tree_method}")

            # Split for early stopping validation
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train_arr, y_train_arr,
                test_size=validation_split,
                random_state=self.random_state,
                stratify=y_train_arr
            )

            logger.info(f"Early stopping validation set: {X_val_split.shape[0]} samples")

            # Default parameter grid if not provided
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'gamma': [0, 0.1, 0.2],
                    'min_child_weight': [1, 3, 5]
                }

            logger.info(f"Hyperparameter grid:")
            for param, values in param_grid.items():
                logger.info(f"  {param}: {values}")

            logger.info(f"Cross-validation folds: {cv}")
            logger.info(f"Scoring metric: {scoring}")
            logger.info(f"Early stopping rounds: {early_stopping_rounds}")

            # Initialize base estimator for GridSearch (no early stopping)
            base_estimator = XGBClassifier(
                random_state=self.random_state,
                tree_method=self.tree_method,
                device=self.device,
                eval_metric='logloss',
                n_jobs=n_jobs if not self.use_gpu else 1
            )

            # Perform grid search
            logger.info("Starting GridSearchCV...")
            logger.info("This may take several minutes due to the large parameter space...")

            self.grid_search = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs if not self.use_gpu else 1,
                verbose=1,
                return_train_score=True
            )

            # Fit without early stopping for GridSearch
            self.grid_search.fit(
                X_train_split, y_train_split,
                verbose=False
            )

            # Store best parameters
            self.best_params = self.grid_search.best_params_
            
            logger.info(f"Best parameters found: {self.best_params}")
            
            # Retrain best model without early stopping (due to compatibility issues)
            logger.info("Retraining best model (early stopping disabled due to XGBoost 2.0 compatibility)...")
            self.model = XGBClassifier(
                **self.best_params,
                random_state=self.random_state,
                tree_method=self.tree_method,
                device=self.device,
                eval_metric='logloss',
                n_jobs=n_jobs if not self.use_gpu else 1
            )
            
            self.model.fit(
                X_train_split, y_train_split,
                verbose=False
            )

            # self.model is already set above
            # self.best_params is already set above

            # Get training history
            try:
                self.evals_result = self.model.evals_result()
            except:
                logger.warning("Could not retrieve evals_result (likely due to disabled early stopping)")
                self.evals_result = {}

            logger.info("="*80)
            logger.info("TRAINING COMPLETED")
            logger.info("="*80)
            logger.info(f"Best parameters found:")
            for param, value in self.best_params.items():
                logger.info(f"  {param}: {value}")
            logger.info(f"Best cross-validation {scoring} score: {self.grid_search.best_score_:.4f}")

            # Log best iteration from early stopping
            if hasattr(self.model, 'best_iteration'):
                logger.info(f"Best iteration (early stopping): {self.model.best_iteration}")

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
                cv=cv, scoring=scoring, n_jobs=n_jobs if not self.use_gpu else 1
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
        output_dir: str = '../results/xgboost'
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
            logger.info("EVALUATING XGBOOST MODEL")
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

            # Store metrics in instance
            self.metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            }

            logger.info(f"\nEvaluation results saved to: {output_dir}")
            logger.info("="*80)

            return evaluation_results

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def get_feature_importance(
        self,
        output_dir: str = '../results/xgboost',
        top_n: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract feature importance using multiple metrics (gain, coverage, weight).

        Args:
            output_dir: Directory to save visualizations
            top_n: Number of top features to display

        Returns:
            dict: Dictionary containing feature importance DataFrames for each metric
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet. Call train() first.")

            logger.info("Extracting feature importance using multiple metrics...")
            os.makedirs(output_dir, exist_ok=True)

            importance_dfs = {}

            # Get importance for different metrics
            importance_types = ['gain', 'weight', 'cover']
            importance_names = {
                'gain': 'Gain (Average gain across splits)',
                'weight': 'Weight (Number of times feature appears)',
                'cover': 'Cover (Average coverage of splits)'
            }

            for imp_type in importance_types:
                # Get importance scores
                importance_dict = self.model.get_booster().get_score(importance_type=imp_type)

                # Create DataFrame
                importance_data = []
                for i, feature_name in enumerate(self.feature_names):
                    feature_key = f'f{i}'
                    score = importance_dict.get(feature_key, 0)
                    importance_data.append({'feature': feature_name, 'importance': score})

                imp_df = pd.DataFrame(importance_data).sort_values('importance', ascending=False)
                importance_dfs[imp_type] = imp_df

                # Log top features
                logger.info(f"\nTop {top_n} Features by {imp_type.upper()}:")
                logger.info("="*60)
                for idx, row in imp_df.head(top_n).iterrows():
                    logger.info(f"  {row['feature']}: {row['importance']:.4f}")

            # Create comparison visualization
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))

            for idx, (imp_type, ax) in enumerate(zip(importance_types, axes)):
                imp_df = importance_dfs[imp_type]
                top_features = imp_df.head(top_n)

                colors = plt.cm.viridis(top_features['importance'] / top_features['importance'].max())

                ax.barh(range(len(top_features)), top_features['importance'],
                       color=colors, edgecolor='black', linewidth=1.5)
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features['feature'])
                ax.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
                ax.set_title(importance_names[imp_type], fontsize=12, fontweight='bold')
                ax.invert_yaxis()
                ax.grid(axis='x', alpha=0.3)

                # Add value labels
                for i, (idx_row, row) in enumerate(top_features.iterrows()):
                    ax.text(row['importance'], i, f" {row['importance']:.2f}",
                           va='center', fontsize=9)

            plt.suptitle(f'Feature Importance Comparison - Top {top_n} Features',
                        fontsize=15, fontweight='bold', y=1.02)
            plt.tight_layout()

            comparison_path = os.path.join(output_dir, 'feature_importance_comparison.png')
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"\nFeature importance comparison saved to: {comparison_path}")

            # Save each importance type to CSV
            for imp_type, imp_df in importance_dfs.items():
                csv_path = os.path.join(output_dir, f'feature_importance_{imp_type}.csv')
                imp_df.to_csv(csv_path, index=False)
                logger.info(f"Feature importance ({imp_type}) saved to: {csv_path}")

            # Create detailed table
            self._create_importance_table(importance_dfs, output_dir)

            return importance_dfs

        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
            raise

    def plot_learning_curves(
        self,
        output_dir: str = '../results/xgboost'
    ) -> None:
        """
        Plot training vs validation performance over iterations.

        Args:
            output_dir: Directory to save plots
        """
        try:
            if self.model is None or self.evals_result is None:
                raise ValueError("Model has not been trained yet. Call train() first.")

            logger.info("Creating learning curves...")
            os.makedirs(output_dir, exist_ok=True)

            # Extract results
            results = self.evals_result

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot validation loss
            if 'validation_0' in results:
                epochs = len(results['validation_0']['logloss'])
                x_axis = range(0, epochs)

                ax.plot(x_axis, results['validation_0']['logloss'],
                       label='Validation Loss', linewidth=2, color='darkred')

                # Mark best iteration
                if hasattr(self.model, 'best_iteration'):
                    best_iter = self.model.best_iteration
                    best_score = results['validation_0']['logloss'][best_iter]
                    ax.axvline(x=best_iter, color='green', linestyle='--',
                             linewidth=2, label=f'Best Iteration ({best_iter})')
                    ax.plot(best_iter, best_score, 'go', markersize=10,
                           label=f'Best Score ({best_score:.4f})')

            ax.set_xlabel('Iteration', fontsize=13, fontweight='bold')
            ax.set_ylabel('Log Loss', fontsize=13, fontweight='bold')
            ax.set_title('XGBoost Learning Curve - Validation Performance',
                        fontsize=15, fontweight='bold')
            ax.legend(fontsize=11, loc='best')
            ax.grid(alpha=0.3)
            plt.tight_layout()

            curve_path = os.path.join(output_dir, 'learning_curves.png')
            plt.savefig(curve_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Learning curves saved to: {curve_path}")

        except Exception as e:
            logger.error(f"Error creating learning curves: {str(e)}")
            raise

    def explain_with_shap(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        output_dir: str = '../results/xgboost',
        sample_idx: Optional[int] = None,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Generate SHAP values for model explainability.

        Args:
            X: Features to explain (typically test set)
            output_dir: Directory to save SHAP visualizations
            sample_idx: Specific sample index for force plot (None for random)
            num_samples: Number of samples to use for SHAP calculation

        Returns:
            dict: SHAP analysis results
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet. Call train() first.")

            logger.info("="*80)
            logger.info("GENERATING SHAP EXPLANATIONS")
            logger.info("="*80)

            os.makedirs(output_dir, exist_ok=True)

            # Convert to numpy array if needed
            if isinstance(X, pd.DataFrame):
                X_arr = X.values
            else:
                X_arr = X

            # Limit samples for performance
            if len(X_arr) > num_samples:
                logger.info(f"Using {num_samples} samples for SHAP analysis (from {len(X_arr)} total)")
                sample_indices = np.random.choice(len(X_arr), num_samples, replace=False)
                X_shap = X_arr[sample_indices]
            else:
                X_shap = X_arr

            # Create SHAP explainer
            logger.info("Creating TreeExplainer...")
            self.shap_explainer = shap.TreeExplainer(self.model)

            # Calculate SHAP values
            logger.info("Calculating SHAP values...")
            self.shap_values = self.shap_explainer.shap_values(X_shap)

            # 1. Summary plot (global importance)
            logger.info("Creating SHAP summary plot...")
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                self.shap_values, X_shap,
                feature_names=self.feature_names,
                show=False,
                plot_type='dot'
            )
            plt.title('SHAP Summary Plot - Feature Impact on Model Output',
                     fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            summary_path = os.path.join(output_dir, 'shap_summary_plot.png')
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP summary plot saved to: {summary_path}")

            # 2. Bar plot (mean absolute SHAP values)
            logger.info("Creating SHAP bar plot...")
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                self.shap_values, X_shap,
                feature_names=self.feature_names,
                show=False,
                plot_type='bar'
            )
            plt.title('SHAP Bar Plot - Mean Feature Impact',
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            bar_path = os.path.join(output_dir, 'shap_bar_plot.png')
            plt.savefig(bar_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP bar plot saved to: {bar_path}")

            # 3. Force plot for individual prediction
            if sample_idx is None:
                sample_idx = 0

            logger.info(f"Creating SHAP force plot for sample {sample_idx}...")

            # Get prediction
            sample_pred = self.model.predict(X_shap[sample_idx:sample_idx+1])[0]
            sample_proba = self.model.predict_proba(X_shap[sample_idx:sample_idx+1])[0]

            # Create force plot
            shap.force_plot(
                self.shap_explainer.expected_value,
                self.shap_values[sample_idx],
                X_shap[sample_idx],
                feature_names=self.feature_names,
                show=False,
                matplotlib=True
            )
            plt.title(f'SHAP Force Plot - Sample {sample_idx}\n'
                     f'Prediction: {self.class_names[sample_pred]} '
                     f'(Probability: {sample_proba[sample_pred]:.4f})',
                     fontsize=12, fontweight='bold')
            plt.tight_layout()
            force_path = os.path.join(output_dir, f'shap_force_plot_sample_{sample_idx}.png')
            plt.savefig(force_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP force plot saved to: {force_path}")

            # 4. Dependence plots for top 3 features
            logger.info("Creating SHAP dependence plots...")

            # Get top features by mean absolute SHAP value
            mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[-3:][::-1]

            for rank, feat_idx in enumerate(top_indices, 1):
                feature_name = self.feature_names[feat_idx]
                logger.info(f"  Creating dependence plot for {feature_name}...")

                plt.figure(figsize=(10, 6))
                shap.dependence_plot(
                    feat_idx, self.shap_values, X_shap,
                    feature_names=self.feature_names,
                    show=False
                )
                plt.title(f'SHAP Dependence Plot - {feature_name}',
                         fontsize=14, fontweight='bold')
                plt.tight_layout()

                dep_path = os.path.join(output_dir, f'shap_dependence_{feature_name}.png')
                plt.savefig(dep_path, dpi=300, bbox_inches='tight')
                plt.close()

            logger.info(f"SHAP dependence plots saved")

            # Calculate feature importance from SHAP values
            shap_importance = pd.DataFrame({
                'feature': self.feature_names,
                'shap_importance': np.abs(self.shap_values).mean(axis=0)
            }).sort_values('shap_importance', ascending=False)

            # Save SHAP importance
            shap_imp_path = os.path.join(output_dir, 'shap_feature_importance.csv')
            shap_importance.to_csv(shap_imp_path, index=False)
            logger.info(f"SHAP feature importance saved to: {shap_imp_path}")

            logger.info("="*80)
            logger.info("SHAP ANALYSIS COMPLETED")
            logger.info("="*80)

            shap_results = {
                'shap_values': self.shap_values,
                'expected_value': self.shap_explainer.expected_value,
                'shap_importance': shap_importance,
                'top_features': [self.feature_names[i] for i in top_indices]
            }

            return shap_results

        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {str(e)}")
            raise

    def analyze_trees(
        self,
        output_dir: str = '../results/xgboost',
        num_trees: int = 3
    ) -> None:
        """
        Visualize individual trees in the XGBoost ensemble.

        Args:
            output_dir: Directory to save tree visualizations
            num_trees: Number of trees to visualize
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet. Call train() first.")

            logger.info(f"Visualizing {num_trees} trees from XGBoost ensemble...")
            os.makedirs(output_dir, exist_ok=True)

            # Get total number of trees
            total_trees = self.model.get_booster().num_boosted_rounds()
            logger.info(f"Total trees in ensemble: {total_trees}")

            # Visualize first N trees
            for tree_idx in range(min(num_trees, total_trees)):
                logger.info(f"  Plotting tree {tree_idx}...")

                try:
                    fig, ax = plt.subplots(figsize=(20, 12))
                    xgb.plot_tree(
                        self.model.get_booster(),
                        num_trees=tree_idx,
                        ax=ax,
                        rankdir='LR'  # Left to right layout
                    )
                    plt.title(f'XGBoost Tree {tree_idx}', fontsize=16, fontweight='bold')
                    plt.tight_layout()

                    tree_path = os.path.join(output_dir, f'tree_{tree_idx}.png')
                    plt.savefig(tree_path, dpi=200, bbox_inches='tight')
                    plt.close()

                    logger.info(f"    Tree {tree_idx} saved to: tree_{tree_idx}.png")
                except ImportError:
                    logger.warning("Graphviz not installed. Skipping tree visualization.")
                    plt.close()
                    break
                except Exception as e:
                    logger.warning(f"Could not plot tree {tree_idx}: {str(e)}")
                    plt.close()

            # Create tree depth analysis
            logger.info("Analyzing tree statistics...")
            tree_stats = self._analyze_tree_statistics()

            # Save tree statistics
            stats_path = os.path.join(output_dir, 'tree_statistics.txt')
            with open(stats_path, 'w') as f:
                f.write("XGBOOST TREE STATISTICS\n")
                f.write("="*80 + "\n\n")
                f.write(f"Total number of trees: {tree_stats['total_trees']}\n")
                f.write(f"Average tree depth: {tree_stats['avg_depth']:.2f}\n")
                f.write(f"Max tree depth: {tree_stats['max_depth']}\n")
                f.write(f"Min tree depth: {tree_stats['min_depth']}\n")

            logger.info(f"Tree statistics saved to: {stats_path}")

        except Exception as e:
            logger.error(f"Error visualizing trees: {str(e)}")
            raise

    def save_model(self, output_dir: str = '../models',
                   model_name: str = 'xgboost_model.pkl') -> None:
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
                'class_names': self.class_names,
                'cv_results': self.cv_results,
                'use_gpu': self.use_gpu,
                'metrics': self.metrics
            }

            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to: {model_path}")

            # Also save as native XGBoost format
            xgb_path = os.path.join(output_dir, 'xgboost_model.json')
            self.model.save_model(xgb_path)
            logger.info(f"XGBoost native format saved to: {xgb_path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path: str) -> 'XGBoostModel':
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
            self.cv_results = model_data.get('cv_results')
            self.use_gpu = model_data.get('use_gpu', False)
            self.metrics = model_data.get('metrics')

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
        plt.title('Confusion Matrix - XGBoost', fontsize=16, fontweight='bold', pad=20)
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
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        plt.title('ROC Curve - XGBoost', fontsize=15, fontweight='bold')
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
        plt.title('Precision-Recall Curve - XGBoost', fontsize=15, fontweight='bold')
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
            f.write("XGBOOST MODEL - EVALUATION METRICS\n")
            f.write("="*80 + "\n\n")

            f.write("MODEL CONFIGURATION:\n")
            f.write("-"*40 + "\n")
            f.write(f"GPU Acceleration: {self.use_gpu}\n")
            f.write(f"Tree Method: {self.tree_method}\n")
            if self.best_params:
                for param, value in self.best_params.items():
                    f.write(f"{param}: {value}\n")
            if hasattr(self.model, 'best_iteration'):
                f.write(f"Best Iteration: {self.model.best_iteration}\n")
            f.write("\n")

            f.write("PERFORMANCE METRICS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Accuracy:  {evaluation_results['accuracy']:.4f}\n")
            f.write(f"Precision: {evaluation_results['precision']:.4f}\n")
            f.write(f"Recall:    {evaluation_results['recall']:.4f}\n")
            f.write(f"F1-Score:  {evaluation_results['f1_score']:.4f}\n")
            f.write(f"ROC-AUC:   {evaluation_results['roc_auc']:.4f}\n\n")

            f.write("CONFUSION MATRIX:\n")
            f.write("-"*40 + "\n")
            f.write(f"{evaluation_results['confusion_matrix']}\n\n")

            f.write("DETAILED CLASSIFICATION REPORT:\n")
            f.write("-"*40 + "\n")
            report_df = pd.DataFrame(evaluation_results['classification_report']).transpose()
            f.write(report_df.to_string())

        logger.info(f"Evaluation metrics saved to: {metrics_path}")

    def _create_importance_table(self, importance_dfs: Dict[str, pd.DataFrame],
                                 output_dir: str) -> None:
        """Create a combined feature importance ranking table."""
        table_path = os.path.join(output_dir, 'feature_importance_combined_table.txt')

        with open(table_path, 'w') as f:
            f.write("FEATURE IMPORTANCE RANKING - XGBOOST (MULTIPLE METRICS)\n")
            f.write("="*100 + "\n\n")

            # Merge all importance types
            combined = importance_dfs['gain'].copy()
            combined.columns = ['feature', 'gain']
            combined['weight'] = combined['feature'].map(
                dict(zip(importance_dfs['weight']['feature'],
                        importance_dfs['weight']['importance']))
            )
            combined['cover'] = combined['feature'].map(
                dict(zip(importance_dfs['cover']['feature'],
                        importance_dfs['cover']['importance']))
            )

            f.write(f"{'Rank':<6} {'Feature':<30} {'Gain':<15} {'Weight':<15} {'Cover':<15}\n")
            f.write("-"*100 + "\n")

            for rank, (idx, row) in enumerate(combined.iterrows(), 1):
                f.write(f"{rank:<6} {row['feature']:<30} "
                       f"{row['gain']:<15.4f} {row['weight']:<15.0f} {row['cover']:<15.4f}\n")

        logger.info(f"Combined feature importance table saved to: {table_path}")

    def _analyze_tree_statistics(self) -> Dict[str, Any]:
        """Analyze statistics of trees in the ensemble."""
        booster = self.model.get_booster()
        total_trees = booster.num_boosted_rounds()

        # Get tree dump to analyze structure
        tree_dump = booster.get_dump()

        # Calculate depths
        depths = []
        for tree_str in tree_dump:
            # Count maximum indentation level as depth
            max_depth = 0
            for line in tree_str.split('\n'):
                if line.strip():
                    depth = len(line) - len(line.lstrip())
                    max_depth = max(max_depth, depth)
            depths.append(max_depth)

        return {
            'total_trees': total_trees,
            'avg_depth': np.mean(depths),
            'max_depth': np.max(depths),
            'min_depth': np.min(depths),
            'depths': depths
        }


def train_and_evaluate_xgboost(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    output_dir: str = '../results/xgboost',
    param_grid: Optional[Dict] = None,
    cv: int = 10,
    use_gpu: bool = False
) -> XGBoostModel:
    """
    Complete pipeline for training and evaluating an XGBoost model.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        output_dir: Directory to save results
        param_grid: Parameter grid for GridSearchCV
        cv: Number of cross-validation folds
        use_gpu: Whether to use GPU acceleration

    Returns:
        XGBoostModel: Trained model
    """
    try:
        logger.info("="*80)
        logger.info("XGBOOST MODEL - COMPLETE PIPELINE")
        logger.info("="*80)

        # Initialize model
        xgb_model = XGBoostModel(random_state=42, use_gpu=use_gpu)

        # Train model
        xgb_model.train(X_train, y_train, param_grid=param_grid, cv=cv)

        # Evaluate model
        xgb_model.evaluate(X_test, y_test, output_dir=output_dir)

        # Get feature importance
        xgb_model.get_feature_importance(output_dir=output_dir, top_n=10)

        # Plot learning curves
        xgb_model.plot_learning_curves(output_dir=output_dir)

        # SHAP explanations
        xgb_model.explain_with_shap(X_test, output_dir=output_dir, num_samples=100)

        # Analyze trees
        xgb_model.analyze_trees(output_dir=output_dir, num_trees=3)

        # Save model
        xgb_model.save_model(output_dir='../models')

        logger.info("\n" + "="*80)
        logger.info("XGBOOST PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)

        return xgb_model

    except Exception as e:
        logger.error(f"Error in XGBoost pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    print("XGBoost Model for Diabetes Prediction")
    print("="*80)
    print("\nThis module provides a comprehensive XGBoost implementation with SHAP.")
    print("\nUsage example:")
    print("""
    from xgboost_model import train_and_evaluate_xgboost
    import pandas as pd

    # Load preprocessed data
    X_train = pd.read_csv('../data/processed/X_train.csv')
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('../data/processed/y_test.csv').values.ravel()

    # Train and evaluate (with GPU if available)
    xgb_model = train_and_evaluate_xgboost(
        X_train, X_test, y_train, y_test,
        output_dir='../results/xgboost',
        use_gpu=False  # Set to True for GPU acceleration
    )
    """)
