"""
Train all models with GPU acceleration (where supported)

This script trains:
1. Decision Tree (CPU only - no GPU support in scikit-learn)
2. Random Forest (CPU only - no GPU support in scikit-learn)
3. XGBoost (GPU accelerated with tree_method='gpu_hist')
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.decision_tree_model import train_and_evaluate_decision_tree
from src.random_forest_model import train_and_evaluate_random_forest
from src.xgboost_model import train_and_evaluate_xgboost
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Train all models with GPU acceleration where supported."""
    print("="*80)
    print("TRAINING ALL MODELS WITH GPU ACCELERATION")
    print("="*80)
    print()
    print("NOTE:")
    print("- XGBoost: Will use GPU (NVIDIA RTX 4070)")
    print("- Random Forest: CPU only (scikit-learn doesn't support GPU)")
    print("- Decision Tree: CPU only (scikit-learn doesn't support GPU)")
    print()
    print("="*80)
    print()
    
    try:
        # Check if processed data exists
        data_dir = 'data/processed'
        X_train_path = os.path.join(data_dir, 'X_train.csv')
        X_test_path = os.path.join(data_dir, 'X_test.csv')
        y_train_path = os.path.join(data_dir, 'y_train.csv')
        y_test_path = os.path.join(data_dir, 'y_test.csv')
        
        if not all(os.path.exists(p) for p in [X_train_path, X_test_path, y_train_path, y_test_path]):
            logger.error("Processed data not found. Please run preprocessing.py first.")
            logger.info("Run: python src/preprocessing.py")
            return
        
        # Load data
        logger.info("Loading preprocessed data...")
        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        y_train = pd.read_csv(y_train_path).values.ravel()
        y_test = pd.read_csv(y_test_path).values.ravel()
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        print()
        
        # Simplified parameter grids for faster training
        simple_param_grid_xgb = {
            'n_estimators': [100, 200],
            'max_depth': [5, 7],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
        }
        
        simple_param_grid_rf = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        }
        
        simple_param_grid_dt = {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        
        # 1. Train XGBoost with GPU
        print("\n" + "="*80)
        print("STEP 1/3: Training XGBoost with GPU Acceleration")
        print("="*80)
        logger.info("Using NVIDIA RTX 4070 with CUDA 13.0")
        logger.info("Tree method: hist, Device: cuda")
        print()
        
        xgb_model = train_and_evaluate_xgboost(
            X_train, X_test, y_train, y_test,
            output_dir='results/xgboost',
            param_grid=simple_param_grid_xgb,
            cv=5,  # Reduced for faster training
            use_gpu=True  # ENABLE GPU!
        )
        logger.info("✓ XGBoost model trained and saved (GPU accelerated)")
        
        # 2. Train Random Forest (CPU)
        print("\n" + "="*80)
        print("STEP 2/3: Training Random Forest (CPU)")
        print("="*80)
        logger.info("Note: scikit-learn Random Forest does not support GPU")
        print()
        
        from src.random_forest_model import RandomForestModel
        rf_model = RandomForestModel(random_state=42)
        rf_model.train(X_train, y_train, param_grid=simple_param_grid_rf, cv=5, n_jobs=-1)
        rf_model.evaluate(X_test, y_test, output_dir='results/random_forest')
        rf_model.save_model(output_dir='models')
        logger.info("✓ Random Forest model trained and saved (CPU)")
        
        # 3. Train Decision Tree (CPU)
        print("\n" + "="*80)
        print("STEP 3/3: Training Decision Tree (CPU)")
        print("="*80)
        logger.info("Note: scikit-learn Decision Tree does not support GPU")
        print()
        
        from src.decision_tree_model import DecisionTreeModel
        dt_model = DecisionTreeModel(random_state=42)
        dt_model.train(X_train, y_train, param_grid=simple_param_grid_dt, cv=5, n_jobs=-1)
        dt_model.evaluate(X_test, y_test, output_dir='results/decision_tree')
        dt_model.save_model(output_dir='models')
        logger.info("✓ Decision Tree model trained and saved (CPU)")
        
        print("\n" + "="*80)
        print("ALL MODELS TRAINED SUCCESSFULLY!")
        print("="*80)
        print()
        print("Models saved to:")
        print("  - models/xgboost_model.pkl (GPU accelerated)")
        print("  - models/random_forest_model.pkl")
        print("  - models/decision_tree_model.pkl")
        print()
        print("Results saved to:")
        print("  - results/xgboost/")
        print("  - results/random_forest/")
        print("  - results/decision_tree/")
        print()
        print("You can now start the backend server:")
        print("  python app.py")
        print()
        
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        raise


if __name__ == "__main__":
    main()
