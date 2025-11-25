"""
Extract product names from reviews using GPT-4o-mini (BATCH MODE - FAST)
"""

import pandas as pd
import pickle
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv
from tqdm import tqdm
import json

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

print("="*60)
print("EXTRACTING PRODUCT NAMES (BATCH MODE)")
print("="*60)

# Load data
print("\n1. Loading dataset...")
df = pd.read_csv('data/raw/fake_reviews_dataset.csv')
genuine_reviews = df[df['label'] == 'OR'].copy()

# PROCESS ONLY A SUBSET - Much faster!
SAMPLE_SIZE = 3000  # Process 3,000 reviews (~$0.10 cost)
genuine_reviews = genuine_reviews.sample(n=SAMPLE_SIZE, random_state=42)
print(f"   Processing {len(genuine_reviews):,} reviews (sampled)")

# Load metadata
cache_dir = Path('.cache')
metadata_path = cache_dir / 'reviews_metadata.pkl'

with open(metadata_path, 'rb') as f:
    metadata = pickle.load(f)

# Create index mapping for fast lookup
review_to_metadata_idx = {}
for idx, item in enumerate(metadata):
    review_to_metadata_idx[item['text'][:100]] = idx  # Use first 100 chars as key

def extract_batch_product_names(reviews_batch):
    """Extract multiple product names in one API call - MUCH FASTER"""
    
    # Create batch prompt
    batch_items = []
    for i, (text, category) in enumerate(reviews_batch):
        batch_items.append(f"{i+1}. Category: {category}\nReview: {text[:200]}...")
    
    prompt = f"""Extract product names from these reviews. For each review, return just the product name or "Generic [Category]" if unclear.

{chr(10).join(batch_items)}

Return as JSON array: [{{"id": 1, "product": "Product Name"}}, ...]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON
        # Remove markdown if present
        if '```json' in result_text:
            result_text = result_text.split('```json')[1].split('```')[0]
        elif '```' in result_text:
            result_text = result_text.split('```')[1].split('```')[0]
        
        results = json.loads(result_text)
        return results
        
    except Exception as e:
        print(f"Batch error: {e}")
        # Fallback: generic names
        return [{"id": i+1, "product": f"Generic {cat.replace('_5', '').replace('_', ' ')}"} 
                for i, (_, cat) in enumerate(reviews_batch)]

# Process in batches of 10 reviews per API call
print("\n2. Extracting product names in batches...")
BATCH_SIZE = 10
total_batches = len(genuine_reviews) // BATCH_SIZE

processed_count = 0

for i in tqdm(range(0, len(genuine_reviews), BATCH_SIZE)):
    batch_df = genuine_reviews.iloc[i:i+BATCH_SIZE]
    
    # Prepare batch
    reviews_batch = list(zip(batch_df['text_'], batch_df['category']))
    
    # Extract names for batch
    results = extract_batch_product_names(reviews_batch)
    
    # Update metadata
    for j, (_, row) in enumerate(batch_df.iterrows()):
        review_key = row['text_'][:100]
        if review_key in review_to_metadata_idx:
            metadata_idx = review_to_metadata_idx[review_key]
            
            # Find matching result
            product_name = next(
                (r['product'] for r in results if r['id'] == j+1),
                f"Generic {row['category'].replace('_5', '').replace('_', ' ')}"
            )
            
            metadata[metadata_idx]['product_name'] = product_name
            processed_count += 1
    
    # Save progress every 100 reviews
    if processed_count % 100 == 0:
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

# Final save
print("\n3. Saving metadata...")
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)

print(f"\n✅ Processed {processed_count:,} reviews")
print(f"Estimated cost: ~${processed_count * 0.0001:.2f}")