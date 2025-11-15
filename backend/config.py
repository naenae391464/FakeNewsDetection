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
    CACHE_DIR = BASE_DIR / ".cache"
    
    # Create directories if they don't exist
    for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, CACHE_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Model Settings
    SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
    ROBERTA_MODEL = os.getenv("ROBERTA_MODEL", "roberta-base")
    
    # API Settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))

config = Config()