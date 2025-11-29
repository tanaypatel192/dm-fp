# XGBoost Model Results

This directory contains all outputs from the XGBoost model training, evaluation, and explainability analysis.

## Files Generated

### Model Evaluation
- **evaluation_metrics.txt** - Complete performance metrics including:
  - Model configuration (GPU status, tree method, hyperparameters)
  - Best iteration from early stopping
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - Confusion matrix
  - Detailed classification report
- **confusion_matrix.png** - Visual confusion matrix heatmap
- **roc_curve.png** - ROC curve with AUC score
- **precision_recall_curve.png** - Precision-Recall curve

### Feature Importance (Multiple Metrics)
- **feature_importance_comparison.png** - 3-panel comparison of importance metrics:
  - **Gain**: Average gain across splits (most commonly used)
  - **Weight**: Number of times feature appears in trees
  - **Cover**: Average coverage of splits using the feature
- **feature_importance_gain.csv** - Gain-based importance scores
- **feature_importance_weight.csv** - Weight-based importance scores
- **feature_importance_cover.csv** - Cover-based importance scores
- **feature_importance_combined_table.txt** - All metrics in one table

### Learning Curves
- **learning_curves.png** - Training vs validation loss over iterations
  - Shows convergence behavior
  - Marks best iteration from early stopping
  - Helps identify overfitting

### SHAP Explainability
- **shap_summary_plot.png** - Global feature impact visualization
  - Shows which features matter most
  - Direction of impact (positive/negative)
  - Impact distribution across samples
- **shap_bar_plot.png** - Mean absolute SHAP values
  - Quick overview of feature importance
  - Based on actual model predictions
- **shap_force_plot_sample_[N].png** - Individual prediction explanation
  - Shows how each feature contributed
  - Base value vs final prediction
  - Positive and negative contributions
- **shap_dependence_[feature].png** - Feature effect plots (top 3 features)
  - Shows how feature values affect predictions
  - Reveals non-linear relationships
  - Includes interaction effects
- **shap_feature_importance.csv** - SHAP-based importance ranking

### Tree Visualization
- **tree_0.png, tree_1.png, tree_2.png** - Visualizations of first 3 trees
  - Shows decision paths
  - Split conditions and leaf values
  - Helps understand model logic
- **tree_statistics.txt** - Tree ensemble statistics
  - Total number of trees
  - Average, max, and min tree depths
  - Structural analysis

## XGBoost Specific Features

### Gradient Boosting
- Sequential ensemble where each tree corrects errors of previous trees
- Learning rate controls contribution of each tree
- Early stopping prevents overfitting

### Multiple Importance Metrics
1. **Gain** (default): Total gain from all splits using the feature
2. **Weight**: How many times feature is used for splitting
3. **Cover**: Average number of samples affected by splits

### SHAP Integration
- **Model-agnostic explanations**: Understand any prediction
- **Global insights**: See overall feature patterns
- **Local explanations**: Explain individual predictions
- **Interaction detection**: Find feature dependencies

### Early Stopping
- Monitors validation performance during training
- Stops when performance plateaus
- Prevents overfitting automatically
- Saves computation time

## Usage

These files are automatically generated when you run the XGBoost training pipeline:

```python
from xgboost_model import train_and_evaluate_xgboost

xgb_model = train_and_evaluate_xgboost(
    X_train, X_test, y_train, y_test,
    output_dir='../results/xgboost',
    use_gpu=False  # Set to True for GPU acceleration
)
```

## GPU Acceleration

If NVIDIA GPU is available, enable GPU acceleration for faster training:

```python
xgb_model = XGBoostModel(use_gpu=True)
xgb_model.train(X_train, y_train)
```

Benefits:
- 10-100x faster training on large datasets
- Handles larger parameter grids efficiently
- Same results as CPU (deterministic)

## Interpreting SHAP Values

### Summary Plot
- **Y-axis**: Features ranked by importance
- **X-axis**: SHAP value (impact on prediction)
- **Color**: Feature value (red=high, blue=low)
- **Each dot**: One sample

### Force Plot
- **Red arrows**: Push prediction toward positive class
- **Blue arrows**: Push prediction toward negative class
- **Arrow length**: Magnitude of effect
- **Base value**: Average model prediction

### Dependence Plot
- **X-axis**: Feature value
- **Y-axis**: SHAP value (impact)
- **Color**: Interaction feature
- Shows how predictions change with feature values

## Key Insights

The XGBoost analysis provides:
1. **Gradient Boosting Power**: Sequential error correction
2. **Multiple Importance Views**: Gain, weight, and cover
3. **Learning Behavior**: Convergence and early stopping
4. **SHAP Explainability**: Why model makes predictions
5. **Tree Structure**: Understand decision logic
6. **Non-linear Effects**: Partial dependence via SHAP

All visualizations are saved at 300 DPI for publication quality.
Trees are saved at 200 DPI due to their complexity and size.
