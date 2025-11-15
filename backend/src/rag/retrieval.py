from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path
from functools import lru_cache
import os

CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# Global embedder - loaded once
_embedder = None

def get_embedder():
    """Singleton pattern for embedder"""
    global _embedder
    if _embedder is None:
        print("📦 Loading embedder (one-time)...")
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedder ready!")
    return _embedder

@lru_cache(maxsize=1)
def load_faiss_index():
    """Load pre-built FAISS index from cache"""
    index_path = CACHE_DIR / "reviews.index"
    metadata_path = CACHE_DIR / "reviews_metadata.pkl"
    
    if not index_path.exists():
        raise FileNotFoundError("FAISS index not built yet. Run build_index.py first")
    
    index = faiss.read_index(str(index_path))
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return index, metadata

def get_recommendations(query_text: str, category: str = None, top_k: int = 5):
    """Fast retrieval with pre-built index"""
    embedder = get_embedder()
    index, metadata = load_faiss_index()
    
    # Encode query - FAST (50ms for MiniLM)
    query_embedding = embedder.encode([query_text], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    # Search - VERY FAST (<1ms for 100k vectors)
    distances, indices = index.search(query_embedding, top_k * 2)  # Get more, filter by category
    
    # Filter and format results
    results = []
    for idx, score in zip(indices[0], distances[0]):
        review_data = metadata[idx]
        
        # Filter by category if specified
        if category and review_data.get('category') != category:
            continue
        
        results.append({
            'text': review_data['text'],
            'rating': review_data['rating'],
            'category': review_data['category'],
            'similarity_score': float(score)
        })
        
        if len(results) >= top_k:
            break
    
    return results