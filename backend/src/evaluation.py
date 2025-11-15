"""
Model Evaluation Module
Handles comprehensive evaluation metrics for fake review detection models
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, cohen_kappa_score, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

class ModelEvaluator:
    """Comprehensive model evaluation and comparison"""
    
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def evaluate(self, y_true, y_pred, y_pred_proba=None, model_name="model"):
        """
        Comprehensive evaluation of a single model
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional, for ROC/AUC)
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        # Convert labels if needed (CG/OR to 1/0)
        y_true_binary = self._convert_labels(y_true)
        y_pred_binary = self._convert_labels(y_pred)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
            'cohens_kappa': cohen_kappa_score(y_true_binary, y_pred_binary)
        }
        
        # Add ROC-AUC if probabilities provided
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true_binary, y_pred_proba)
        
        # Store results
        self.results[model_name] = metrics
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Evaluation Results: {model_name}")
        print(f"{'='*60}")
        for metric, value in metrics.items():
            print(f"{metric:20s}: {value:.4f}")
        
        # Check if meets target thresholds
        self._check_thresholds(metrics, model_name)
        
        return metrics
    
    def _convert_labels(self, labels):
        """Convert CG/OR labels to 1/0"""
        if isinstance(labels, pd.Series):
            labels = labels.values
        
        # If already numeric, return as is
        if isinstance(labels[0], (int, np.integer, float, np.floating)):
            return labels
        
        # Convert string labels
        return np.where(labels == 'CG', 1, 0)
    
    def _check_thresholds(self, metrics, model_name):
        """Check if model meets target thresholds"""
        print(f"\nThreshold Check:")
        
        # Target: ≥97% Accuracy
        if metrics['accuracy'] >= 0.97:
            print(f"  ✓ Accuracy: {metrics['accuracy']:.2%} (≥97% target met)")
        else:
            print(f"  ✗ Accuracy: {metrics['accuracy']:.2%} (below 97% target)")
        
        # Target: κ > 0.7
        if metrics['cohens_kappa'] > 0.7:
            print(f"  ✓ Cohen's Kappa: {metrics['cohens_kappa']:.4f} (>0.7 target met)")
        else:
            print(f"  ✗ Cohen's Kappa: {metrics['cohens_kappa']:.4f} (below 0.7 target)")
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name="model"):
        """Plot and save confusion matrix"""
        y_true_binary = self._convert_labels(y_true)
        y_pred_binary = self._convert_labels(y_pred)
        
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Original', 'Computer Generated'],
                    yticklabels=['Original', 'Computer Generated'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add accuracy on plot
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        plt.text(0.5, -0.15, f'Accuracy: {accuracy:.2%}', 
                ha='center', transform=plt.gca().transAxes, fontsize=12)
        
        filename = f'{model_name}_confusion_matrix_{self.timestamp}.png'
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Confusion matrix saved: {filename}")
        return cm
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name="model"):
        """Plot and save ROC curve"""
        y_true_binary = self._convert_labels(y_true)
        
        fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_proba)
        roc_auc = roc_auc_score(y_true_binary, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        filename = f'{model_name}_roc_curve_{self.timestamp}.png'
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ ROC curve saved: {filename}")
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name="model"):
        """Plot and save Precision-Recall curve"""
        y_true_binary = self._convert_labels(y_true)
        
        precision, recall, thresholds = precision_recall_curve(y_true_binary, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.grid(alpha=0.3)
        
        filename = f'{model_name}_pr_curve_{self.timestamp}.png'
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Precision-Recall curve saved: {filename}")
    
    def compare_models(self):
        """Compare all evaluated models"""
        if not self.results:
            print("No models evaluated yet!")
            return None
        
        df_results = pd.DataFrame(self.results).T
        df_results = df_results.round(4)
        
        print(f"\n{'='*60}")
        print("Model Comparison")
        print(f"{'='*60}")
        print(df_results.to_string())
        
        # Save to CSV
        filename = f'model_comparison_{self.timestamp}.csv'
        df_results.to_csv(self.results_dir / filename)
        print(f"\n✓ Comparison saved: {filename}")
        
        return df_results
    
    def plot_model_comparison(self):
        """Create comparison bar chart for all models"""
        if not self.results:
            print("No models evaluated yet!")
            return
        
        df = pd.DataFrame(self.results).T
        
        # Plot comparison
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'cohens_kappa', 'roc_auc']
        
        for idx, metric in enumerate(metrics):
            if metric in df.columns:
                ax = axes[idx]
                df[metric].plot(kind='bar', ax=ax, color='steelblue')
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_ylabel('Score')
                ax.set_ylim([0, 1])
                ax.axhline(y=0.97, color='r', linestyle='--', alpha=0.5, label='Target (97%)')
                ax.grid(alpha=0.3)
                ax.legend()
        
        plt.tight_layout()
        filename = f'models_comparison_{self.timestamp}.png'
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparison chart saved: {filename}")
    
    def generate_report(self, y_true, y_pred, model_name="model"):
        """Generate detailed classification report"""
        y_true_binary = self._convert_labels(y_true)
        y_pred_binary = self._convert_labels(y_pred)
        
        report = classification_report(
            y_true_binary, y_pred_binary, 
            target_names=['Original', 'Computer Generated'],
            digits=4
        )
        
        print(f"\n{'='*60}")
        print(f"Classification Report: {model_name}")
        print(f"{'='*60}")
        print(report)
        
        # Save report
        filename = f'{model_name}_classification_report_{self.timestamp}.txt'
        with open(self.results_dir / filename, 'w') as f:
            f.write(f"Classification Report: {model_name}\n")
            f.write(f"{'='*60}\n")
            f.write(report)
        
        print(f"✓ Report saved: {filename}")