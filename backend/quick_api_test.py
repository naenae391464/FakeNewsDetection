from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

# CRITICAL: Add CORS middleware BEFORE any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your best model
model = joblib.load('models/saved_models/logistic_regression.pkl')

class Review(BaseModel):
    text: str

@app.get("/")
async def root():
    return {
        "message": "Fake Review Detection API",
        "status": "running",
        "endpoints": {
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }

@app.post("/predict")
async def predict(review: Review):
    prediction = model.predict([review.text])[0]
    probability = model.predict_proba([review.text])[0]
    
    return {
        "is_fake": bool(prediction),
        "confidence": float(max(probability)),
        "label": "Computer Generated" if prediction else "Original"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)