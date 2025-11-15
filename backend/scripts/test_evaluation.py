"""Test the evaluation module"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import ModelEvaluator
import pandas as pd
import numpy as np

# Load processed data
print("Loading data...")
df = pd.read_csv('data/processed/reviews_with_features.csv')

# Create a simple baseline (for testing evaluation module)
# Just predict based on review length
y_true = df['label']
y_pred = np.where(df['word_count'] < 67, 'CG', 'OR')  # Simple threshold
y_pred_proba = 1 - (df['word_count'] / df['word_count'].max())  # Fake probabilities

# Initialize evaluator
evaluator = ModelEvaluator()

# Evaluate
metrics = evaluator.evaluate(y_true, y_pred, y_pred_proba, model_name="Length_Baseline")

# Generate visualizations
evaluator.plot_confusion_matrix(y_true, y_pred, "Length_Baseline")
evaluator.plot_roc_curve(y_true, y_pred_proba, "Length_Baseline")
evaluator.plot_precision_recall_curve(y_true, y_pred_proba, "Length_Baseline")

# Generate detailed report
evaluator.generate_report(y_true, y_pred, "Length_Baseline")

print("\n✅ Evaluation module test complete!")