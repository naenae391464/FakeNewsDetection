from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import sys
sys.path.append('..')

from src.ml_models.classical import load_cached_model
from src.rag.retrieval import get_recommendations

app = FastAPI(title="Fake Review Detection API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache models on startup for speed
@app.on_event("startup")
async def load_models():
    global classifier, embedder
    print("🚀 Loading models into memory...")
    classifier = load_cached_model("logistic_regression")  # Fastest
    # embedder loaded lazily in retrieval.py
    print("✅ Models loaded!")

class ReviewRequest(BaseModel):
    text: str
    category: Optional[str] = "Electronics"

class DetectionResponse(BaseModel):
    is_fake: bool
    confidence: float
    model_used: str
    processing_time_ms: float

class RecommendationResponse(BaseModel):
    alternatives: List[dict]
    reasoning: str

@app.post("/api/detect", response_model=DetectionResponse)
async def detect_fake_review(review: ReviewRequest):
    import time
    start = time.time()
    
    # Use cached model - FAST
    prediction = classifier.predict([review.text])[0]
    proba = classifier.predict_proba([review.text])[0]
    
    processing_time = (time.time() - start) * 1000
    
    return DetectionResponse(
        is_fake=bool(prediction),
        confidence=float(max(proba)),
        model_used="Logistic Regression (TF-IDF)",
        processing_time_ms=round(processing_time, 2)
    )

@app.post("/api/recommend", response_model=RecommendationResponse)
async def recommend_alternatives(review: ReviewRequest):
    # Only call if review is fake
    alternatives = get_recommendations(
        query_text=review.text,
        category=review.category,
        top_k=5
    )
    
    return RecommendationResponse(
        alternatives=alternatives,
        reasoning="Similar products with verified positive reviews"
    )

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "models_loaded": True}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Development only
        workers=1  # Single worker for development
    )