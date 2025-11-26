# Random Forest Model Results

This directory contains all outputs from the Random Forest model training and evaluation.

## Files Generated

### Model Evaluation
- **evaluation_metrics.txt** - Detailed performance metrics (accuracy, precision, recall, F1, ROC-AUC, OOB score)
- **confusion_matrix.png** - Visual representation of the confusion matrix
- **roc_curve.png** - ROC curve showing model performance
- **precision_recall_curve.png** - Precision-Recall curve

### Feature Importance
- **feature_importance_top_10.png** - Bar chart of top 10 most important features with error bars
- **feature_importance.csv** - Complete feature importance data with standard deviations
- **feature_importance_table.txt** - Formatted ranking table of all features

### Partial Dependence Plots
- **partial_dependence_[feature_name].png** - Individual PD plots for top 3 features
- **partial_dependence_combined.png** - Combined view of all partial dependence plots

### Out-of-Bag (OOB) Analysis
- **oob_error_analysis.png** - OOB error vs number of trees graph
- **oob_analysis.txt** - Detailed OOB error metrics for different tree counts

### Tree Diversity Analysis
- **tree_diversity_analysis.png** - 4-panel visualization showing:
  - Distribution of tree depths
  - Distribution of number of leaves
  - Distribution of features used per tree
  - Distribution of pairwise tree agreement
- **tree_diversity_metrics.txt** - Detailed diversity statistics

### Model Comparison
- **rf_vs_dt_comparison.png** - Bar chart comparing Random Forest vs Decision Tree
- **rf_vs_dt_comparison.txt** - Detailed metric-by-metric comparison with improvements

## Random Forest Specific Features

### Out-of-Bag (OOB) Score
- Measures model performance using samples not selected during bootstrap sampling
- Provides unbiased estimate without need for separate validation set
- Useful for determining optimal number of trees

### Tree Diversity
- Analyzes how different the trees in the forest are from each other
- Lower pairwise agreement = higher diversity = better generalization
- Tracks feature usage, tree depth, and structural variations

### Partial Dependence
- Shows marginal effect of features on predictions
- Helps understand feature-target relationships
- Useful for model interpretation and debugging

## Usage

These files are automatically generated when you run the Random Forest training pipeline:

```python
from random_forest_model import train_and_evaluate_random_forest

rf_model = train_and_evaluate_random_forest(
    X_train, X_test, y_train, y_test,
    output_dir='../results/random_forest'
)
```

## Key Insights

The Random Forest analysis provides:
1. **Ensemble Power**: Shows improvement over single decision tree
2. **Feature Interactions**: Partial dependence reveals non-linear relationships
3. **Model Stability**: OOB analysis confirms consistent performance
4. **Diversity Metrics**: Ensures trees are learning different patterns

All visualizations are saved at 300 DPI for publication quality.
