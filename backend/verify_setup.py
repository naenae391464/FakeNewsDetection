import pandas as pd
import numpy as np
from pathlib import Path
import sys

print("=" * 50)
print("VERIFYING SETUP")
print("=" * 50)

print("\n✓ Checking package imports...")
try:
    import sklearn
    import nltk
    import transformers
    import sentence_transformers
    import faiss
    import fastapi
    import openai
    print("✓ All packages imported successfully!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

print("\n✓ Checking dataset...")
dataset_path = Path("data/raw/fake_reviews_dataset.csv")

if not dataset_path.exists():
    print(f"✗ Dataset not found at {dataset_path}")
    print("  Download from: https://osf.io/tyue9/")
    print("  Save as: backend/data/raw/fake_reviews_dataset.csv")
    sys.exit(1)

df = pd.read_csv(dataset_path)

print(f"✓ Dataset loaded successfully!")
print(f"  Total rows: {len(df):,}")
print(f"  Columns: {df.columns.tolist()}")

if 'label' in df.columns:
    print(f"\n  Class distribution:")
    print(f"    Real (0): {(df['label'] == 0).sum():,}")
    print(f"    Fake (1): {(df['label'] == 1).sum():,}")
    
    if len(df) == 40000 and (df['label'] == 0).sum() == 20000:
        print("  ✓ Dataset is properly balanced!")
    else:
        print("  ⚠ Warning: Dataset may not be the correct version")

if 'category' in df.columns:
    print(f"\n  Categories ({df['category'].nunique()}):")
    for cat, count in df['category'].value_counts().items():
        print(f"    {cat}: {count:,}")

if 'text' in df.columns:
    word_counts = df['text'].str.split().str.len()
    print(f"\n  Review length statistics:")
    print(f"    Mean: {word_counts.mean():.1f} words")
    print(f"    Min: {word_counts.min()} words")
    print(f"    Max: {word_counts.max()} words")

print("\n" + "=" * 50)
print("SETUP VERIFICATION COMPLETE!")
print("=" * 50)
print("\nNext steps:")
print("1. Start with notebooks/01_eda.ipynb for exploratory analysis")
print("2. Train baseline models with Week 1-2 tasks")
print("3. Build API endpoints when models are ready")