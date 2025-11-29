"""
Simple script to fix metrics loading in app.py
"""
import re

# Read the file
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the metrics loading section
old_pattern = r"""                    # Load metrics if available
                    cv_results = model_data\.get\('cv_results', \{\}\)
                    if cv_results:
                        model_metrics\[model_name\] = \{
                            'cv_score': cv_results\.get\('mean_cv_score', 0\.0\),
                            'cv_std': cv_results\.get\('std_cv_score', 0\.0\)
                        \}"""

new_code = """                    # Load metrics if available
                    saved_metrics = model_data.get('metrics', {})
                    if saved_metrics:
                        model_metrics[model_name] = {
                            'accuracy': saved_metrics.get('accuracy', 0.0),
                            'precision': saved_metrics.get('precision', 0.0),
                            'recall': saved_metrics.get('recall', 0.0),
                            'f1_score': saved_metrics.get('f1_score', 0.0),
                            'roc_auc': saved_metrics.get('roc_auc', 0.0)
                        }
                    else:
                        # Fallback to cv_results if metrics not available
                        cv_results = model_data.get('cv_results', {})
                        if cv_results:
                            model_metrics[model_name] = {
                                'accuracy': 0.0,
                                'precision': 0.0,
                                'recall': 0.0,
                                'f1_score': cv_results.get('mean_cv_score', 0.0),
                                'roc_auc': 0.0
                            }"""

# Replace
content = re.sub(old_pattern, new_code, content)

# Write back
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Metrics loading fixed in app.py")
