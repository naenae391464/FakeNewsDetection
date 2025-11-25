"""
Test all three saved models to see which detects fake reviews best
"""

import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

print("="*60)
print("TESTING ALL SAVED MODELS")
print("="*60)

# Load models
models = {
    'Logistic Regression': joblib.load('models/saved_models/logistic_regression.pkl'),
    'Naive Bayes': joblib.load('models/saved_models/naive_bayes.pkl'),
    'SVM Linear': joblib.load('models/saved_models/svm_linear.pkl')
}

# Test reviews
test_reviews = [
    # Obviously fake reviews
    "This is absolutely the best product I have ever purchased in my entire life! The quality is simply outstanding and exceeds all of my expectations in every single way imaginable. I cannot recommend this highly enough to everyone who is looking for something perfect. You will definitely not regret buying this amazing product at all. Five stars is not even close to being enough to express how wonderful and incredible this is!",
    
    "Amazing product! Best purchase ever! Highly recommend! Perfect quality! Outstanding! Five stars! Everyone buy this now! Incredible! Fantastic! Love it!",
    
    "Perfect perfect perfect! Everyone needs to buy this right now! Best ever made! Exceeded all expectations! Cannot recommend highly enough! Five stars is not enough! Outstanding quality! Amazing! Incredible!",
    
    # Real-sounding reviews
    "Decent product overall. Works as described. Had some minor issues with the setup - the instructions could be clearer. Quality is good for the price point. Shipping was fast. Would recommend if you're on a budget.",
    
    "Good sound quality for the price. Battery lasts about 8 hours. Only complaint is it's a bit heavy and the bass could be better. Overall satisfied with the purchase.",
]

labels = [1, 1, 1, 0, 0]  # 1 = Fake, 0 = Real
label_names = ['FAKE', 'FAKE', 'FAKE', 'REAL', 'REAL']

print("\nTesting reviews:\n")

for model_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print('='*60)
    
    predictions = model.predict(test_reviews)
    probabilities = model.predict_proba(test_reviews)
    
    correct = 0
    for i, (review, pred, prob, true_label, label_name) in enumerate(zip(test_reviews, predictions, probabilities, labels, label_names)):
        pred_label = "FAKE" if pred == 1 else "REAL"
        confidence = max(prob) * 100
        
        is_correct = "✓" if pred == true_label else "✗"
        if pred == true_label:
            correct += 1
        
        print(f"\nReview {i+1}: {label_name}")
        print(f"  Preview: {review[:80]}...")
        print(f"  Prediction: {pred_label} ({confidence:.1f}% confidence) {is_correct}")
    
    accuracy = (correct / len(test_reviews)) * 100
    print(f"\n  Overall Accuracy: {correct}/{len(test_reviews)} ({accuracy:.0f}%)")

print("\n" + "="*60)
print("TESTING COMPLETE")
print("="*60)