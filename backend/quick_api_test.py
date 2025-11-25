from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
from pathlib import Path
from typing import Optional
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Week 1 Model
print("Loading Logistic Regression model...")
model = joblib.load('models/saved_models/logistic_regression.pkl')
print("✅ Model loaded")

class Review(BaseModel):
    text: str
    product_name: Optional[str] = None
    category: Optional[str] = None

@app.get("/")
async def root():
    return {
        "message": "Fake Review Detection API with RAG Explanations",
        "status": "running",
        "model": "Logistic Regression + RAG",
        "accuracy": "93.47%"
    }

@app.post("/predict")
async def predict(review: Review):
    """Detect if review is fake/genuine WITH RAG evidence"""
    
    # Predict
    prediction = model.predict([review.text])[0]
    probability = model.predict_proba([review.text])[0]
    confidence = max(probability)
    is_fake = bool(prediction)
    
    # Get RAG evidence
    try:
        from src.rag.retrieval import (
            get_evidence, 
            generate_rag_explanation,
            analyze_review_patterns
        )
        
        # Retrieve similar examples
        evidence = get_evidence(review.text, is_fake, top_k=5)
        
        # Generate AI explanation
        rag_explanation = generate_rag_explanation(
            review.text, 
            is_fake, 
            confidence, 
            evidence
        )
        
        # Analyze patterns
        patterns = analyze_review_patterns(review.text)
        
        return {
            "is_fake": is_fake,
            "confidence": float(confidence),
            "label": "Computer Generated" if is_fake else "Original",
            "model": "Logistic Regression + RAG",
            "rag_explanation": rag_explanation,
            "evidence": evidence,
            "patterns": patterns
        }
        
    except Exception as e:
        print(f"RAG error: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback without RAG
        return {
            "is_fake": is_fake,
            "confidence": float(confidence),
            "label": "Computer Generated" if is_fake else "Original",
            "model": "Logistic Regression"
        }

if __name__ == "__main__":
    # DIAGNOSTIC TEST
    print("\n" + "="*60)
    print("DIAGNOSTIC TEST")
    print("="*60)
    
    test_cases = [
        ("The product is very good. The product works well. I would recommend this product.", "FAKE"),
        ("Bought this for my home office. Setup took about 20 minutes. Worth the $45.", "REAL")
    ]
    
    for text, expected in test_cases:
        prediction = model.predict([text])[0]
        probability = model.predict_proba([text])[0]
        
        is_fake = bool(prediction)
        predicted_label = "FAKE" if is_fake else "REAL"
        confidence = max(probability) * 100
        
        status = "✅ CORRECT" if predicted_label == expected else "❌ WRONG"
        
        print(f"\nTest: {expected}")
        print(f"  Predicted: {predicted_label} ({confidence:.1f}%) {status}")
    
    print("\n" + "="*60)
    print("Ready to serve!")
    print("="*60 + "\n")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)