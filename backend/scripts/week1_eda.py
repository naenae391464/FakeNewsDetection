import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
sns.set_style("whitegrid")
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

# Load data
print("Loading dataset...")
df = pd.read_csv("data/raw/fake_reviews_dataset.csv")

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Basic statistics
print("\n" + "="*50)
print("BASIC STATISTICS")
print("="*50)

print(f"\nClass Distribution:")
print(df['label'].value_counts())

print(f"\nCategory Distribution:")
print(df['category'].value_counts())

# Visualizations
print("\nGenerating visualizations...")

# 1. Class distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='label')
plt.title('Distribution of Real vs Fake Reviews')
plt.xlabel('Label (CG=Computer Generated, OR=Original)')
plt.ylabel('Count')
plt.savefig('results/01_class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Review length distribution
df['word_count'] = df['text_'].str.split().str.len()  # Fixed: text_ not text

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='label', y='word_count')
plt.title('Review Length Distribution: Original vs Computer Generated')
plt.xlabel('Label (CG=Computer Generated, OR=Original)')
plt.ylabel('Word Count')
plt.savefig('results/02_length_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Category distribution by label
plt.figure(figsize=(14, 8))
category_counts = df.groupby(['category', 'label']).size().unstack()
category_counts.plot(kind='bar', stacked=False, figsize=(14, 8))
plt.title('Category Distribution by Label')
plt.xlabel('Category')
plt.ylabel('Count')
plt.legend(['Computer Generated', 'Original'])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/03_category_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Rating distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='rating', hue='label')
plt.title('Rating Distribution: Original vs Computer Generated Reviews')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.legend(['Computer Generated', 'Original'])
plt.savefig('results/04_rating_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ Visualizations saved to results/ folder")

# Summary statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)

summary = df.groupby('label').agg({
    'word_count': ['mean', 'std', 'min', 'max'],
    'rating': ['mean', 'std']
}).round(2)

print(summary)

# Save processed data
print("\nSaving processed data...")
df.to_csv('data/processed/reviews_with_features.csv', index=False)

print("\n✅ Week 1 EDA Complete!")
print(f"\nGenerated files:")
print("  - results/01_class_distribution.png")
print("  - results/02_length_distribution.png")
print("  - results/03_category_distribution.png")
print("  - results/04_rating_distribution.png")
print("  - data/processed/reviews_with_features.csv")