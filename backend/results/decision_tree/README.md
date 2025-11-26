# Decision Tree Model Results

This directory contains all outputs from the Decision Tree model training and evaluation.

## Files Generated

### Model Evaluation
- **evaluation_metrics.txt** - Detailed performance metrics (accuracy, precision, recall, F1, ROC-AUC)
- **confusion_matrix.png** - Visual representation of the confusion matrix
- **roc_curve.png** - ROC curve showing model performance
- **precision_recall_curve.png** - Precision-Recall curve

### Tree Visualization
- **decision_tree_visualization.png** - Full decision tree visualization
- **decision_tree_simplified.png** - Simplified tree (max depth 4) for readability

### Feature Importance
- **feature_importance_all.png** - Bar chart of all feature importance scores
- **feature_importance_top_10.png** - Top 10 most important features
- **feature_importance.csv** - Feature importance data in CSV format

### Decision Rules
- **decision_rules.txt** - Raw text representation of decision tree rules
- **decision_rules_formatted.txt** - Formatted if-then rules for easy interpretation

## Usage

These files are automatically generated when you run the Decision Tree training pipeline:

```python
from decision_tree_model import train_and_evaluate_decision_tree

dt_model = train_and_evaluate_decision_tree(
    X_train, X_test, y_train, y_test,
    output_dir='../results/decision_tree'
)
```

All visualizations are saved at 300 DPI for publication quality.
