from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load your best model
model = joblib.load('models/saved_models/logistic_regression.pkl')

class Review(BaseModel):
    text: str

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