import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = BASE_DIR / "models" / "saved_models"
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Model Settings - OPTIMIZED FOR SPEED
    # Use smaller, faster models
    SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"  # Fast, 80MB only
    ROBERTA_MODEL = "roberta-base"  # 500MB, much faster than large
    
    # Use quantized/distilled models for production
    USE_QUANTIZATION = True  # 4x faster inference
    
    # FAISS Settings
    FAISS_INDEX_TYPE = "Flat"  # Fastest for <1M vectors
    
    # Caching - CRITICAL FOR SPEED
    CACHE_EMBEDDINGS = True  # Don't recompute
    CACHE_DIR = BASE_DIR / ".cache"
    
    # API Settings
    MAX_WORKERS = 4  # For parallel processing
    BATCH_SIZE = 32  # Optimal for CPU
    
    # OpenAI Settings (since you're using paid)
    OPENAI_MODEL = "gpt-4o-mini"  # Fast and cheap
    OPENAI_MAX_TOKENS = 500
    OPENAI_TEMPERATURE = 0.3

config = Config()