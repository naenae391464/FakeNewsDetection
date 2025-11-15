"""
Week 2: Train Classical Machine Learning Models
Baseline models: Naive Bayes, Logistic Regression, SVM
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib
import time

from src.evaluation import ModelEvaluator

print("="*60)
print("WEEK 2: CLASSICAL ML MODEL TRAINING")
print("="*60)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('data/processed/reviews_with_features.csv')
print(f"   Loaded {len(df):,} reviews")

# Prepare data
X = df['text_']
y = df['label']

# Convert labels to binary (CG=1, OR=0)
y_binary = np.where(y == 'CG', 1, 0)

# Train-test split (80-20)
print("\n2. Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
print(f"   Training samples: {len(X_train):,}")
print(f"   Testing samples: {len(X_test):,}")

# Initialize evaluator
evaluator = ModelEvaluator()

# ==================== NAIVE BAYES ====================
print("\n" + "="*60)
print("TRAINING: Naive Bayes")
print("="*60)

start_time = time.time()

nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=2
    )),
    ('clf', MultinomialNB(alpha=0.1))
])

nb_pipeline.fit(X_train, y_train)
train_time = time.time() - start_time

# Predict
y_pred_nb = nb_pipeline.predict(X_test)
y_pred_proba_nb = nb_pipeline.predict_proba(X_test)[:, 1]

print(f"Training time: {train_time:.2f} seconds")

# Evaluate
evaluator.evaluate(y_test, y_pred_nb, y_pred_proba_nb, model_name="Naive_Bayes")
evaluator.plot_confusion_matrix(y_test, y_pred_nb, "Naive_Bayes")
evaluator.plot_roc_curve(y_test, y_pred_proba_nb, "Naive_Bayes")
evaluator.generate_report(y_test, y_pred_nb, "Naive_Bayes")

# Save model
print("\nSaving model...")
joblib.dump(nb_pipeline, 'models/saved_models/naive_bayes.pkl')
print("✓ Model saved: models/saved_models/naive_bayes.pkl")

# ==================== LOGISTIC REGRESSION ====================
print("\n" + "="*60)
print("TRAINING: Logistic Regression")
print("="*60)

start_time = time.time()

lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=2,
        sublinear_tf=True
    )),
    ('clf', LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver='saga',
        n_jobs=-1,
        random_state=42
    ))
])

lr_pipeline.fit(X_train, y_train)
train_time = time.time() - start_time

# Predict
y_pred_lr = lr_pipeline.predict(X_test)
y_pred_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]

print(f"Training time: {train_time:.2f} seconds")

# Evaluate
evaluator.evaluate(y_test, y_pred_lr, y_pred_proba_lr, model_name="Logistic_Regression")
evaluator.plot_confusion_matrix(y_test, y_pred_lr, "Logistic_Regression")
evaluator.plot_roc_curve(y_test, y_pred_proba_lr, "Logistic_Regression")
evaluator.generate_report(y_test, y_pred_lr, "Logistic_Regression")

# Save model
print("\nSaving model...")
joblib.dump(lr_pipeline, 'models/saved_models/logistic_regression.pkl')
print("✓ Model saved: models/saved_models/logistic_regression.pkl")

# ==================== SVM ====================
print("\n" + "="*60)
print("TRAINING: Support Vector Machine (Linear)")
print("="*60)
print("Note: SVM training may take a few minutes...")

start_time = time.time()

svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=2
    )),
    ('clf', SVC(
        kernel='linear',
        C=1.0,
        probability=True,  # Enable probability estimates
        random_state=42
    ))
])

svm_pipeline.fit(X_train, y_train)
train_time = time.time() - start_time

# Predict
y_pred_svm = svm_pipeline.predict(X_test)
y_pred_proba_svm = svm_pipeline.predict_proba(X_test)[:, 1]

print(f"Training time: {train_time:.2f} seconds")

# Evaluate
evaluator.evaluate(y_test, y_pred_svm, y_pred_proba_svm, model_name="SVM_Linear")
evaluator.plot_confusion_matrix(y_test, y_pred_svm, "SVM_Linear")
evaluator.plot_roc_curve(y_test, y_pred_proba_svm, "SVM_Linear")
evaluator.generate_report(y_test, y_pred_svm, "SVM_Linear")

# Save model
print("\nSaving model...")
joblib.dump(svm_pipeline, 'models/saved_models/svm_linear.pkl')
print("✓ Model saved: models/saved_models/svm_linear.pkl")

# ==================== MODEL COMPARISON ====================
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

comparison_df = evaluator.compare_models()
evaluator.plot_model_comparison()

print("\n" + "="*60)
print("✅ WEEK 2 TRAINING COMPLETE!")
print("="*60)
print("\nModels trained and saved:")
print("  1. Naive Bayes")
print("  2. Logistic Regression")
print("  3. SVM (Linear)")
print("\nAll evaluation metrics and visualizations saved to results/")
print("\nNext: Week 3 - Transformer model training")