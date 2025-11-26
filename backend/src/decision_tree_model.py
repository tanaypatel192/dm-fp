"""
Decision Tree Model for Diabetes Prediction

This module implements a comprehensive Decision Tree classifier with:
- Hyperparameter tuning using GridSearchCV
- Detailed evaluation metrics
- Tree visualization
- Feature importance analysis
- Rule extraction
- Prediction explanations

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
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DecisionTreeModel:
    """
    A comprehensive Decision Tree classifier for diabetes prediction.

    This class provides methods for training, evaluation, visualization,
    feature importance analysis, rule extraction, and prediction explanation.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the DecisionTreeModel.

        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.model: Optional[DecisionTreeClassifier] = None
        self.best_params: Optional[Dict] = None
        self.feature_names: Optional[List[str]] = None
        self.class_names: List[str] = ['No Diabetes', 'Diabetes']
        self.cv_results: Optional[Dict] = None
        self.grid_search: Optional[GridSearchCV] = None

        logger.info("DecisionTreeModel initialized")

    def train(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        param_grid: Optional[Dict] = None,
        cv: int = 10,
        scoring: str = 'f1',
        n_jobs: int = -1
    ) -> 'DecisionTreeModel':
        """
        Train the Decision Tree model with hyperparameter tuning using GridSearchCV.

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
            logger.info("TRAINING DECISION TREE MODEL")
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
                    'max_depth': [3, 5, 7, 10, 15, 20],
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 4, 6, 8, 10],
                    'criterion': ['gini', 'entropy']
                }

            logger.info(f"Hyperparameter grid:")
            for param, values in param_grid.items():
                logger.info(f"  {param}: {values}")

            logger.info(f"Cross-validation folds: {cv}")
            logger.info(f"Scoring metric: {scoring}")

            # Initialize base estimator
            base_estimator = DecisionTreeClassifier(random_state=self.random_state)

            # Perform grid search
            logger.info("Starting GridSearchCV...")
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
        output_dir: str = '../results/decision_tree'
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
            logger.info("EVALUATING MODEL")
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

    def visualize_tree(
        self,
        output_dir: str = '../results/decision_tree',
        max_depth: Optional[int] = None,
        figsize: Tuple[int, int] = (25, 20),
        dpi: int = 300
    ) -> None:
        """
        Create and save decision tree visualization.

        Args:
            output_dir: Directory to save the visualization
            max_depth: Maximum depth to visualize (None for full tree)
            figsize: Figure size (width, height)
            dpi: Resolution for saved image
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet. Call train() first.")

            logger.info("Creating decision tree visualization...")
            os.makedirs(output_dir, exist_ok=True)

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Plot tree
            plot_tree(
                self.model,
                feature_names=self.feature_names,
                class_names=self.class_names,
                filled=True,
                rounded=True,
                fontsize=10,
                max_depth=max_depth,
                ax=ax
            )

            plt.title('Decision Tree Visualization', fontsize=20, fontweight='bold', pad=20)
            plt.tight_layout()

            # Save figure
            output_path = os.path.join(output_dir, 'decision_tree_visualization.png')
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close()

            logger.info(f"Decision tree visualization saved to: {output_path}")

            # Also create a simplified version (max_depth=4) for readability
            if max_depth is None and self.model.get_depth() > 4:
                fig, ax = plt.subplots(figsize=(20, 15))
                plot_tree(
                    self.model,
                    feature_names=self.feature_names,
                    class_names=self.class_names,
                    filled=True,
                    rounded=True,
                    fontsize=11,
                    max_depth=4,
                    ax=ax
                )
                plt.title('Decision Tree Visualization (Simplified - Depth 4)',
                         fontsize=20, fontweight='bold', pad=20)
                plt.tight_layout()

                simplified_path = os.path.join(output_dir, 'decision_tree_simplified.png')
                plt.savefig(simplified_path, dpi=dpi, bbox_inches='tight')
                plt.close()

                logger.info(f"Simplified tree visualization saved to: {simplified_path}")

        except Exception as e:
            logger.error(f"Error visualizing tree: {str(e)}")
            raise

    def get_feature_importance(
        self,
        output_dir: str = '../results/decision_tree',
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Extract and visualize feature importance.

        Args:
            output_dir: Directory to save visualizations
            top_n: Number of top features to display in detail

        Returns:
            DataFrame: Feature importance scores
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet. Call train() first.")

            logger.info("Extracting feature importance...")
            os.makedirs(output_dir, exist_ok=True)

            # Get feature importance
            importance = self.model.feature_importances_

            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)

            # Log top features
            logger.info(f"\nTop {top_n} Most Important Features:")
            logger.info("="*60)
            for idx, row in importance_df.head(top_n).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

            # Visualize all features
            plt.figure(figsize=(12, max(8, len(self.feature_names) * 0.3)))
            colors = plt.cm.viridis(importance_df['importance'] / importance_df['importance'].max())

            plt.barh(range(len(importance_df)), importance_df['importance'], color=colors)
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
            plt.ylabel('Features', fontsize=12, fontweight='bold')
            plt.title('Feature Importance - Decision Tree', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()

            output_path = os.path.join(output_dir, 'feature_importance_all.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Feature importance plot saved to: {output_path}")

            # Visualize top N features
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(top_n)
            colors = plt.cm.RdYlGn(top_features['importance'] / top_features['importance'].max())

            plt.barh(range(len(top_features)), top_features['importance'],
                    color=colors, edgecolor='black', linewidth=1.5)
            plt.yticks(range(len(top_features)), top_features['feature'], fontsize=11)
            plt.xlabel('Importance Score', fontsize=13, fontweight='bold')
            plt.ylabel('Features', fontsize=13, fontweight='bold')
            plt.title(f'Top {top_n} Most Important Features', fontsize=15, fontweight='bold')
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

            logger.info(f"Top {top_n} features plot saved to: {top_n_path}")

            # Save to CSV
            csv_path = os.path.join(output_dir, 'feature_importance.csv')
            importance_df.to_csv(csv_path, index=False)
            logger.info(f"Feature importance data saved to: {csv_path}")

            return importance_df

        except Exception as e:
            logger.error(f"Error extracting feature importance: {str(e)}")
            raise

    def extract_rules(
        self,
        output_dir: str = '../results/decision_tree',
        max_depth: Optional[int] = None
    ) -> str:
        """
        Convert decision tree to interpretable if-then rules.

        Args:
            output_dir: Directory to save the rules
            max_depth: Maximum depth for rule extraction

        Returns:
            str: Text representation of the rules
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet. Call train() first.")

            logger.info("Extracting decision rules from tree...")
            os.makedirs(output_dir, exist_ok=True)

            # Get text representation
            tree_rules = export_text(
                self.model,
                feature_names=self.feature_names,
                max_depth=max_depth,
                decimals=3
            )

            # Save to file
            rules_path = os.path.join(output_dir, 'decision_rules.txt')
            with open(rules_path, 'w') as f:
                f.write("DECISION TREE RULES\n")
                f.write("="*80 + "\n\n")
                f.write("Rule Format:\n")
                f.write("  - Each path from root to leaf represents a decision rule\n")
                f.write("  - Numbers in brackets [x, y] represent class distribution\n")
                f.write("  - class: X indicates the predicted class\n\n")
                f.write("="*80 + "\n\n")
                f.write(tree_rules)

            logger.info(f"Decision rules saved to: {rules_path}")

            # Extract and format specific rules
            formatted_rules = self._format_decision_rules()

            # Save formatted rules
            formatted_path = os.path.join(output_dir, 'decision_rules_formatted.txt')
            with open(formatted_path, 'w') as f:
                f.write("FORMATTED DECISION RULES\n")
                f.write("="*80 + "\n\n")
                for i, rule in enumerate(formatted_rules, 1):
                    f.write(f"Rule {i}:\n")
                    f.write(f"{rule}\n\n")

            logger.info(f"Formatted rules saved to: {formatted_path}")

            # Log sample rules
            logger.info(f"\nSample Decision Rules (showing first 3):")
            logger.info("="*60)
            for i, rule in enumerate(formatted_rules[:3], 1):
                logger.info(f"\nRule {i}:")
                logger.info(rule)

            return tree_rules

        except Exception as e:
            logger.error(f"Error extracting rules: {str(e)}")
            raise

    def predict_with_explanation(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        sample_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Make predictions and explain the decision path.

        Args:
            X: Input features (single sample or multiple samples)
            sample_idx: Index of sample to explain (if None, explains first sample)

        Returns:
            dict: Prediction and explanation details
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been trained yet. Call train() first.")

            # Convert to numpy array if needed
            if isinstance(X, pd.DataFrame):
                X_arr = X.values
            else:
                X_arr = X

            # Ensure 2D array
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)

            # Select sample to explain
            if sample_idx is None:
                sample_idx = 0

            if sample_idx >= len(X_arr):
                raise ValueError(f"sample_idx {sample_idx} out of range for data with {len(X_arr)} samples")

            sample = X_arr[sample_idx:sample_idx+1]

            # Make prediction
            prediction = self.model.predict(sample)[0]
            prediction_proba = self.model.predict_proba(sample)[0]

            # Get decision path
            decision_path = self.model.decision_path(sample)
            node_indicator = decision_path.toarray()[0]

            # Get the nodes along the path
            node_indices = np.where(node_indicator)[0]

            # Extract path information
            tree = self.model.tree_
            feature_indices = tree.feature[node_indices]
            threshold_values = tree.threshold[node_indices]

            # Build explanation
            explanation_steps = []
            for i, node_idx in enumerate(node_indices[:-1]):  # Exclude leaf node
                feature_idx = feature_indices[i]
                threshold = threshold_values[i]
                feature_name = self.feature_names[feature_idx]
                feature_value = sample[0, feature_idx]

                if sample[0, feature_idx] <= threshold:
                    direction = "<="
                else:
                    direction = ">"

                step = {
                    'node': node_idx,
                    'feature': feature_name,
                    'feature_value': feature_value,
                    'threshold': threshold,
                    'condition': f"{feature_name} {direction} {threshold:.3f}",
                    'actual_value': feature_value
                }
                explanation_steps.append(step)

            # Create explanation dictionary
            explanation = {
                'sample_idx': sample_idx,
                'prediction': int(prediction),
                'prediction_label': self.class_names[prediction],
                'prediction_probability': {
                    self.class_names[0]: prediction_proba[0],
                    self.class_names[1]: prediction_proba[1]
                },
                'confidence': max(prediction_proba),
                'decision_path': explanation_steps,
                'number_of_nodes_visited': len(node_indices),
                'feature_values': dict(zip(self.feature_names, sample[0]))
            }

            # Log explanation
            logger.info(f"\nPrediction Explanation for Sample {sample_idx}:")
            logger.info("="*80)
            logger.info(f"Prediction: {self.class_names[prediction]}")
            logger.info(f"Confidence: {max(prediction_proba):.4f}")
            logger.info(f"Probability Distribution:")
            for class_name, prob in explanation['prediction_probability'].items():
                logger.info(f"  {class_name}: {prob:.4f}")

            logger.info(f"\nDecision Path ({len(explanation_steps)} steps):")
            for i, step in enumerate(explanation_steps, 1):
                logger.info(f"  Step {i}: {step['condition']}")
                logger.info(f"           (actual value: {step['actual_value']:.3f})")

            return explanation

        except Exception as e:
            logger.error(f"Error making prediction with explanation: {str(e)}")
            raise

    def save_model(self, output_dir: str = '../models', model_name: str = 'decision_tree_model.pkl') -> None:
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
                'cv_results': self.cv_results
            }

            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to: {model_path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path: str) -> 'DecisionTreeModel':
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
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
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
        plt.title('Receiver Operating Characteristic (ROC) Curve',
                 fontsize=15, fontweight='bold')
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
        plt.title('Precision-Recall Curve', fontsize=15, fontweight='bold')
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
            f.write("DECISION TREE MODEL - EVALUATION METRICS\n")
            f.write("="*80 + "\n\n")

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

    def _format_decision_rules(self) -> List[str]:
        """Format decision tree rules into readable if-then format."""
        tree = self.model.tree_
        feature_names = self.feature_names

        def recurse(node, depth, path_conditions):
            indent = "  " * depth
            if tree.feature[node] != -2:  # Not a leaf
                name = feature_names[tree.feature[node]]
                threshold = tree.threshold[node]

                # Left child (<=)
                left_condition = f"{name} <= {threshold:.3f}"
                left_path = path_conditions + [left_condition]
                recurse(tree.children_left[node], depth + 1, left_path)

                # Right child (>)
                right_condition = f"{name} > {threshold:.3f}"
                right_path = path_conditions + [right_condition]
                recurse(tree.children_right[node], depth + 1, right_path)
            else:
                # Leaf node - create rule
                class_idx = np.argmax(tree.value[node][0])
                class_name = self.class_names[class_idx]
                samples = int(tree.n_node_samples[node])
                probability = tree.value[node][0][class_idx] / samples

                rule = "IF " + " AND ".join(path_conditions)
                rule += f"\nTHEN Prediction = {class_name}"
                rule += f"\n     (Samples: {samples}, Confidence: {probability:.4f})"
                rules.append(rule)

        rules = []
        recurse(0, 0, [])
        return rules


def train_and_evaluate_decision_tree(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    output_dir: str = '../results/decision_tree',
    param_grid: Optional[Dict] = None,
    cv: int = 10
) -> DecisionTreeModel:
    """
    Complete pipeline for training and evaluating a Decision Tree model.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        output_dir: Directory to save results
        param_grid: Parameter grid for GridSearchCV
        cv: Number of cross-validation folds

    Returns:
        DecisionTreeModel: Trained model
    """
    try:
        logger.info("="*80)
        logger.info("DECISION TREE MODEL - COMPLETE PIPELINE")
        logger.info("="*80)

        # Initialize model
        dt_model = DecisionTreeModel(random_state=42)

        # Train model
        dt_model.train(X_train, y_train, param_grid=param_grid, cv=cv)

        # Evaluate model
        dt_model.evaluate(X_test, y_test, output_dir=output_dir)

        # Visualize tree
        dt_model.visualize_tree(output_dir=output_dir)

        # Get feature importance
        dt_model.get_feature_importance(output_dir=output_dir)

        # Extract rules
        dt_model.extract_rules(output_dir=output_dir)

        # Save model
        dt_model.save_model(output_dir='../models')

        # Example prediction with explanation
        logger.info("\n" + "="*80)
        logger.info("EXAMPLE PREDICTION WITH EXPLANATION")
        logger.info("="*80)
        dt_model.predict_with_explanation(X_test, sample_idx=0)

        logger.info("\n" + "="*80)
        logger.info("DECISION TREE PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)

        return dt_model

    except Exception as e:
        logger.error(f"Error in decision tree pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    print("Decision Tree Model for Diabetes Prediction")
    print("="*80)
    print("\nThis module provides a comprehensive Decision Tree implementation.")
    print("\nUsage example:")
    print("""
    from decision_tree_model import train_and_evaluate_decision_tree
    import pandas as pd

    # Load preprocessed data
    X_train = pd.read_csv('../data/processed/X_train.csv')
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('../data/processed/y_test.csv').values.ravel()

    # Train and evaluate
    dt_model = train_and_evaluate_decision_tree(
        X_train, X_test, y_train, y_test,
        output_dir='../results/decision_tree'
    )
    """)
