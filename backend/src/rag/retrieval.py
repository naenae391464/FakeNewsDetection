"""
RAG System - Evidence-Based Explanations
Retrieves similar reviews to justify fake/genuine classification
"""

from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path
from functools import lru_cache
import numpy as np

CACHE_DIR = Path(__file__).parent.parent.parent / ".cache"

_embedder = None

def get_embedder():
    """Singleton pattern for embedder"""
    global _embedder
    if _embedder is None:
        print("Loading embedder (one-time)...")
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedder ready!")
    return _embedder

@lru_cache(maxsize=1)
def load_faiss_index():
    """Load pre-built FAISS index"""
    index_path = CACHE_DIR / "reviews.index"
    metadata_path = CACHE_DIR / "reviews_metadata.pkl"
    
    if not index_path.exists():
        raise FileNotFoundError("FAISS index not found. Run build_faiss_index.py first")
    
    index = faiss.read_index(str(index_path))
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Loaded FAISS index with {len(metadata):,} reviews")
    return index, metadata

def load_fake_reviews_index():
    """Load FAISS index of FAKE reviews for comparison"""
    fake_index_path = CACHE_DIR / "fake_reviews.index"
    fake_metadata_path = CACHE_DIR / "fake_reviews_metadata.pkl"
    
    if not fake_index_path.exists():
        print("Building fake reviews index (one-time)...")
        build_fake_reviews_index()
    
    index = faiss.read_index(str(fake_index_path))
    with open(fake_metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return index, metadata

def build_fake_reviews_index():
    """Build FAISS index from fake reviews (CG labeled)"""
    import pandas as pd
    
    df = pd.read_csv('data/raw/fake_reviews_dataset.csv')
    fake_reviews = df[df['label'] == 'CG'].copy()
    
    print(f"Building index from {len(fake_reviews):,} fake reviews...")
    
    embedder = get_embedder()
    
    # Sample for speed
    fake_reviews = fake_reviews.sample(n=min(5000, len(fake_reviews)), random_state=42)
    
    # Generate embeddings
    texts = fake_reviews['text_'].tolist()
    embeddings = embedder.encode(texts, show_progress_bar=True)
    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)
    
    # Build index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    # Save metadata
    metadata = [{'text': row['text_'], 'rating': row['rating']} 
                for _, row in fake_reviews.iterrows()]
    
    faiss.write_index(index, str(CACHE_DIR / "fake_reviews.index"))
    with open(CACHE_DIR / "fake_reviews_metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    print("✓ Fake reviews index built")

def get_evidence(query_text: str, is_fake: bool, top_k: int = 5):
    """
    Get evidence to support the classification decision
    """
    embedder = get_embedder()
    
    # Encode query
    query_embedding = embedder.encode([query_text], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    if is_fake:
        # Find similar FAKE reviews
        fake_index, fake_metadata = load_fake_reviews_index()
        distances, indices = fake_index.search(query_embedding.astype('float32'), top_k)
        
        similar_examples = []
        for idx, score in zip(indices[0], distances[0]):
            similar_examples.append({
                'text': fake_metadata[idx]['text'],
                'similarity': float(score)
            })
        
        # Also get genuine reviews for contrast
        genuine_index, genuine_metadata = load_faiss_index()
        distances_gen, indices_gen = genuine_index.search(query_embedding.astype('float32'), 3)
        
        genuine_contrast = []
        for idx, score in zip(indices_gen[0], distances_gen[0]):
            genuine_contrast.append({
                'text': genuine_metadata[idx]['text'][:200],
                'rating': genuine_metadata[idx]['rating']
            })
        
        return {
            'similar_examples': similar_examples,
            'genuine_contrast': genuine_contrast,
            'classification': 'fake'
        }
    
    else:
        # Find similar GENUINE reviews
        genuine_index, genuine_metadata = load_faiss_index()
        distances, indices = genuine_index.search(query_embedding.astype('float32'), top_k)
        
        similar_examples = []
        for idx, score in zip(indices[0], distances[0]):
            similar_examples.append({
                'text': genuine_metadata[idx]['text'],
                'rating': genuine_metadata[idx]['rating'],
                'similarity': float(score)
            })
        
        return {
            'similar_examples': similar_examples,
            'classification': 'genuine'
        }

def generate_rag_explanation(query_text: str, is_fake: bool, confidence: float, evidence: dict):
    """Generate GPT-4o-mini explanation using retrieved evidence"""
    try:
        from openai import OpenAI
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        if is_fake:
            examples_text = "\n".join([
                f"{i+1}. \"{ex['text'][:150]}...\" ({ex['similarity']*100:.0f}% similar)"
                for i, ex in enumerate(evidence['similar_examples'][:3])
            ])
            
            prompt = f"""A review was classified as FAKE with {confidence*100:.0f}% confidence.

REVIEW: "{query_text[:200]}"

SIMILAR FAKE PATTERNS FROM TRAINING DATA:
{examples_text}

Write 2-3 sentences explaining what patterns make this review appear fake and why these similar examples support the classification. Be specific and educational."""

        else:
            examples_text = "\n".join([
                f"{i+1}. \"{ex['text'][:150]}...\" ({ex['rating']:.1f}★, {ex['similarity']*100:.0f}% similar)"
                for i, ex in enumerate(evidence['similar_examples'][:3])
            ])
            
            prompt = f"""A review was classified as GENUINE with {confidence*100:.0f}% confidence.

REVIEW: "{query_text[:200]}"

SIMILAR GENUINE REVIEWS FROM TRAINING DATA:
{examples_text}

Write 2-3 sentences explaining what characteristics indicate this is genuine and why these similar examples support the classification. Be specific and educational."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"GPT explanation failed: {e}")
        
        if is_fake:
            return f"This review matches {len(evidence['similar_examples'])} fake review patterns from our training data."
        else:
            return f"This review matches {len(evidence['similar_examples'])} genuine review patterns from our training data."

def analyze_review_patterns(text: str):
    """Analyze linguistic patterns in the review"""
    words = text.lower().split()
    
    # Count repetitions
    word_counts = {}
    for word in words:
        if len(word) > 3:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    repetitive_words = {w: c for w, c in word_counts.items() if c >= 3}
    
    # Detect generic phrases
    generic_phrases = ['the product', 'this item', 'very good', 'excellent', 
                      'highly recommend', 'great quality', 'works well']
    found_generics = [phrase for phrase in generic_phrases if phrase in text.lower()]
    
    # Check for specifics
    has_numbers = any(char.isdigit() for char in text)
    has_dollar = '$' in text
    time_words = ['minutes', 'hours', 'days', 'weeks', 'months', 'years']
    has_timeframe = any(word in text.lower() for word in time_words)
    
    return {
        'repetitive_words': repetitive_words,
        'generic_phrases': found_generics,
        'has_specifics': has_numbers or has_dollar or has_timeframe,
        'word_count': len(words)
    }