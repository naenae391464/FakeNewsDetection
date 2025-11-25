"""
Build FAISS index from genuine reviews for fast similarity search
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path

print("="*60)
print("BUILDING FAISS INDEX FOR RAG SYSTEM")
print("="*60)

# Load data
print("\n1. Loading dataset...")
df = pd.read_csv('data/raw/fake_reviews_dataset.csv')

# Get only genuine reviews
genuine_reviews = df[df['label'] == 'OR'].copy()
print(f"   Found {len(genuine_reviews):,} genuine reviews")

# Initialize embedder (lightweight model)
print("\n2. Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("   Model loaded (dimension: 384)")

# Create embeddings
print("\n3. Generating embeddings (this may take a few minutes)...")
texts = genuine_reviews['text_'].tolist()
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

print(f"   Generated {len(embeddings):,} embeddings")
print(f"   Embedding shape: {embeddings.shape}")

# Normalize for cosine similarity
print("\n4. Normalizing embeddings...")
faiss.normalize_L2(embeddings)

# Build FAISS index
print("\n5. Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity after normalization
index.add(embeddings.astype('float32'))

print(f"   Index built with {index.ntotal:,} vectors")

# Prepare metadata
print("\n6. Preparing metadata...")
metadata = []
for idx, row in genuine_reviews.iterrows():
    metadata.append({
        'text': row['text_'],
        'category': row['category'],
        'rating': row['rating']
    })

# Save everything
print("\n7. Saving to disk...")
cache_dir = Path('.cache')
cache_dir.mkdir(exist_ok=True)

# Save FAISS index
faiss.write_index(index, str(cache_dir / 'reviews.index'))
print(f"   ✓ FAISS index saved: .cache/reviews.index")

# Save metadata
with open(cache_dir / 'reviews_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)
print(f"   ✓ Metadata saved: .cache/reviews_metadata.pkl")

print("\n" + "="*60)
print("✅ FAISS INDEX BUILD COMPLETE!")
print("="*60)
print(f"\nIndex contains {len(metadata):,} genuine reviews")
print(f"Ready for fast similarity search!")