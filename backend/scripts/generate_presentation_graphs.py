"""
Generate all graphs and visualizations for presentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
import joblib
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Create output directory
output_dir = Path('presentation_graphs')
output_dir.mkdir(exist_ok=True)

print("="*60)
print("GENERATING PRESENTATION VISUALIZATIONS")
print("="*60)

# Load data
print("\n1. Loading data and predictions...")
df = pd.read_csv('data/raw/fake_reviews_dataset.csv')
df['label_binary'] = (df['label'] == 'CG').astype(int)

# Load Week 1 models
models = {
    'Logistic Regression': joblib.load('models/saved_models/logistic_regression.pkl'),
    'Naive Bayes': joblib.load('models/saved_models/naive_bayes.pkl'),
    'SVM Linear': joblib.load('models/saved_models/svm_linear.pkl')
}

# Split data (same as training)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df['text_'], df['label_binary'],
    test_size=0.2,
    random_state=42,
    stratify=df['label_binary']
)

print(f"   Test set: {len(X_test):,} samples")

# ============================================================
# GRAPH 1: Model Accuracy Comparison (Bar Chart)
# ============================================================
print("\n2. Creating Model Accuracy Comparison...")

accuracies = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    accuracies[name] = accuracy * 100

# Add RoBERTa (manually - use the value from Colab)
accuracies['RoBERTa'] = 95.75

# Create bar chart
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
bars = ax.bar(accuracies.keys(), accuracies.values(), color=colors, alpha=0.8, edgecolor='black')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Comparison - Test Set Accuracy', fontsize=16, fontweight='bold')
ax.set_ylim([88, 98])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: model_accuracy_comparison.png")
plt.close()

# ============================================================
# GRAPH 2: Confusion Matrices (Subplots)
# ============================================================
print("\n3. Creating Confusion Matrices...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                cbar_kws={'label': 'Count'})
    
    axes[idx].set_title(f'{name}\nAccuracy: {accuracies[name]:.2f}%', 
                       fontsize=14, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=12)
    axes[idx].set_xlabel('Predicted Label', fontsize=12)

# Hide 4th subplot (only 3 models)
axes[3].axis('off')
axes[3].text(0.5, 0.5, f'RoBERTa\n(Week 3)\nAccuracy: 95.75%', 
             ha='center', va='center', fontsize=20, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: confusion_matrices.png")
plt.close()

# ============================================================
# GRAPH 3: ROC Curves
# ============================================================
print("\n4. Creating ROC Curves...")

plt.figure(figsize=(10, 8))

for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, linewidth=2, 
             label=f'{name} (AUC = {roc_auc:.3f})')

# Diagonal line
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
plt.title('ROC Curves - Week 1 Models', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: roc_curves.png")
plt.close()

# ============================================================
# GRAPH 4: Precision-Recall Curves
# ============================================================
print("\n5. Creating Precision-Recall Curves...")

plt.figure(figsize=(10, 8))

for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    
    plt.plot(recall, precision, linewidth=2, label=name)

plt.xlabel('Recall', fontsize=14, fontweight='bold')
plt.ylabel('Precision', fontsize=14, fontweight='bold')
plt.title('Precision-Recall Curves - Week 1 Models', fontsize=16, fontweight='bold')
plt.legend(loc="best", fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: precision_recall_curves.png")
plt.close()

# ============================================================
# GRAPH 5: Classification Report Table
# ============================================================
print("\n6. Creating Classification Report Table...")

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

for idx, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, 
                                   target_names=['Real', 'Fake'],
                                   output_dict=True)
    
    # Convert to DataFrame
    df_report = pd.DataFrame(report).transpose()
    df_report = df_report.round(3)
    
    # Create table
    axes[idx].axis('tight')
    axes[idx].axis('off')
    
    table = axes[idx].table(
        cellText=df_report.values,
        colLabels=df_report.columns,
        rowLabels=df_report.index,
        cellLoc='center',
        loc='center',
        colWidths=[0.15] * len(df_report.columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df_report.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[idx].set_title(f'{name} - Classification Report', 
                       fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(output_dir / 'classification_reports.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: classification_reports.png")
plt.close()

# ============================================================
# GRAPH 6: Dataset Distribution
# ============================================================
print("\n7. Creating Dataset Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Label distribution
label_counts = df['label'].value_counts()
axes[0].pie(label_counts.values, 
           labels=['Real (OR)', 'Fake (CG)'],
           autopct='%1.1f%%',
           colors=['#2ecc71', '#e74c3c'],
           startangle=90,
           explode=(0.05, 0.05))
axes[0].set_title('Dataset Label Distribution', fontsize=14, fontweight='bold')

# Category distribution
category_counts = df['category'].value_counts().head(10)
axes[1].barh(range(len(category_counts)), category_counts.values, color='skyblue', edgecolor='black')
axes[1].set_yticks(range(len(category_counts)))
axes[1].set_yticklabels([cat.replace('_5', '').replace('_', ' ') for cat in category_counts.index])
axes[1].set_xlabel('Number of Reviews', fontsize=12, fontweight='bold')
axes[1].set_title('Top 10 Product Categories', fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'dataset_distribution.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: dataset_distribution.png")
plt.close()

# ============================================================
# GRAPH 7: Review Length Distribution
# ============================================================
print("\n8. Creating Review Length Distribution...")

df['word_count'] = df['text_'].apply(lambda x: len(x.split()))

fig, ax = plt.subplots(figsize=(12, 6))

# Separate by label
real_lengths = df[df['label'] == 'OR']['word_count']
fake_lengths = df[df['label'] == 'CG']['word_count']

ax.hist(real_lengths, bins=50, alpha=0.6, label='Real Reviews', color='green', edgecolor='black')
ax.hist(fake_lengths, bins=50, alpha=0.6, label='Fake Reviews', color='red', edgecolor='black')

ax.set_xlabel('Number of Words', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
ax.set_title('Review Length Distribution by Label', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'review_length_distribution.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: review_length_distribution.png")
plt.close()

# ============================================================
# GRAPH 8: Model Comparison Summary Table
# ============================================================
print("\n9. Creating Model Comparison Summary Table...")

# Create summary data
summary_data = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (y_pred == y_test).mean()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    summary_data.append({
        'Model': name,
        'Accuracy': f'{accuracy*100:.2f}%',
        'Precision': f'{precision:.3f}',
        'Recall': f'{recall:.3f}',
        'F1-Score': f'{f1:.3f}',
        'True Pos': tp,
        'False Pos': fp
    })

# Add RoBERTa
summary_data.append({
    'Model': 'RoBERTa',
    'Accuracy': '95.75%',
    'Precision': '0.958',
    'Recall': '0.957',
    'F1-Score': '0.957',
    'True Pos': '1915',
    'False Pos': '85'
})

df_summary = pd.DataFrame(summary_data)

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('tight')
ax.axis('off')

table = ax.table(
    cellText=df_summary.values,
    colLabels=df_summary.columns,
    cellLoc='center',
    loc='center',
    colWidths=[0.18, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header
for i in range(len(df_summary.columns)):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight RoBERTa row
for i in range(len(df_summary.columns)):
    table[(4, i)].set_facecolor('#fffacd')
    table[(4, i)].set_text_props(weight='bold')

ax.set_title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(output_dir / 'model_summary_table.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: model_summary_table.png")
plt.close()

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("✅ ALL GRAPHS GENERATED!")
print("="*60)
print(f"\nSaved to: {output_dir.absolute()}")
print("\nGenerated files:")
print("  1. model_accuracy_comparison.png - Bar chart of all models")
print("  2. confusion_matrices.png - 2x2 grid of confusion matrices")
print("  3. roc_curves.png - ROC curves for Week 1 models")
print("  4. precision_recall_curves.png - PR curves")
print("  5. classification_reports.png - Detailed metrics tables")
print("  6. dataset_distribution.png - Dataset overview")
print("  7. review_length_distribution.png - Word count analysis")
print("  8. model_summary_table.png - Complete comparison table")
print("\nUse these in your PowerPoint presentation!")
print("="*60)