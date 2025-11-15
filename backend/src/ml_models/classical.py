import pickle
import os
from pathlib import Path
from functools import lru_cache

MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "saved_models"

# Cache models in memory - CRITICAL FOR SPEED
_model_cache = {}

@lru_cache(maxsize=5)
def load_cached_model(model_name: str):
    """Load model once, cache in memory"""
    if model_name in _model_cache:
        return _model_cache[model_name]
    
    model_path = MODELS_DIR / f"{model_name}.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_name} not found")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    _model_cache[model_name] = model
    return model

def train_fast_classifier(X_train, y_train):
    """Train optimized Logistic Regression - FASTEST"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,  # Balance speed vs accuracy
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2,
            sublinear_tf=True  # Faster
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            n_jobs=-1,  # Use all CPU cores
            solver='saga',  # Fastest for large datasets
            C=1.0
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline