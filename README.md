# Fake Review Detection System with RAG-based Evidence Explanations

An end-to-end machine learning system that detects fake product reviews and provides evidence-based explanations using Retrieval-Augmented Generation (RAG).


## Project Overview

This system combines traditional ML, transformer models, and RAG to:
- **Detect** fake product reviews with high accuracy
- **Retrieve** similar review examples using FAISS vector search
- **Explain** predictions with evidence-based reasoning using GPT-4o-mini

**Course:** CSE 573 - Data Mining  
**Dataset:** Fake Reviews Dataset (Salminen et al., 2021) - 40,000 Amazon reviews (50% real, 50% GPT-2 generated)  
**Source:** https://osf.io/tyue9/

---

## Repository Structure

```
FakeNewsDetection/
├── README.md
│
├── backend/                                      # FastAPI + ML models + RAG
│   ├── DATA/                                     # Datasets
│   │   ├── raw/                                  # Original 40K reviews
│   │   └── processed/                            # Cleaned / preprocessed data
│   │
│   ├── evaluations/                              # Evaluation outputs
│   │   ├── complete_presentation_graphs/         # Accuracy, ROC, confusion matrix, summaries
│   │   └── models/                               # Saved model outputs, metrics, logs
│   │
│   ├── scripts/                                  # CODE (training scripts, utility scripts)
│   └── src/                                      # CODE (FastAPI app, ML pipeline, RAG components)
│
└── frontend/                                     # React UI + Frontend Code
```

---

## System Architecture
```
User Input (Product Review)
      ↓
┌──────────────────────────────────┐
│   Preprocessing                  │
│   NLTK + TF-IDF                  │
└──────────────────────────────────┘
      ↓
┌──────────────────────────────────┐
│   Detection Models               │
│   • Week 1: Traditional ML       │
│     (LR, NB, SVM)               │
│   • Week 3: RoBERTa Transformer  │
└──────────────────────────────────┘
      ↓
   Fake Detected?
      ↓
┌──────────────────────────────────┐
│   RAG System (Week 4)            │
│   • FAISS vector search          │
│   • Sentence-BERT embeddings     │
│   • Pattern analysis             │
│   • GPT-4o-mini explanations     │
└──────────────────────────────────┘
      ↓
Evidence-Based Result + Explanation
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Git

### 1. Clone Repository
```bash
git clone .....
cd fake-review-detection
```

### 2. Backend Setup
```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create `backend/.env`:
```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Download Required Files

**Download the dataset:**
1. Visit https://osf.io/tyue9/
2. Download `fake_reviews_dataset.csv`
3. Place in `DATA/raw/`

**Download pre-trained models** (optional):
- Place models in `/backend/models/saved_models/`
- Place RoBERTa in `/backend/models/roberta_fake_review_model/`

### 5. Start Backend Server
```bash
# Make sure you're in /backend/ directory
python quick_api_test.py
```

**Expected output:**
```
Loading RoBERTa model...
RoBERTa loaded on cpu
Loading embedder (one-time)...
Embedder ready!
Loaded FAISS index with 20,216 reviews
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Backend is ready at:** http://localhost:8000

### 6. Frontend Setup (New Terminal)
```bash
# Navigate to frontend
cd /frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

**Expected output:**
```
VITE v5.x.x  ready in xxx ms

➜  Local:   http://localhost:5173/
```

### 7. Open Application

Visit: **http://localhost:5173**


## Training From Scratch

### Week 1: Traditional ML Models
```bash
cd /backend
python scripts/train_week1_models.py
```

**Trains:**
- Logistic Regression (93.47%)
- Naive Bayes (88.97%)
- SVM Linear (93.76%)

**Output:** `/backend/models/saved_models/`

### Week 2: Neural Networks (Skipped)

**Originally planned:** LSTM, GRU, CNN architectures

### Week 3: RoBERTa Transformer

**Use Google Colab (GPU Required):**
1. Open https://colab.research.google.com
2. Runtime → Change runtime → **T4 GPU**
3. Upload dataset from `DATA/raw/`
4. Run `/backend/scripts/train_roberta_colab.py`

**Output:** `/backend/models/roberta_fake_review_model/` (95.75% accuracy)

### Week 4: RAG System Setup

```bash
cd /backend
python scripts/build_faiss_index.py
```

**Output:** FAISS index with 20K genuine reviews for similarity search


## Generate Evaluation Graphs
```bash
cd /backend
python scripts/generate_presentation_graphs.py
```

**Output:** `EVALUATIONS/graphs/` - 8 visualizations (accuracy, confusion matrices, ROC curves, etc.)


## Detailed File Structure
```
fake-review-detection/
├── README.md
│
├── /
│   ├── backend/
│   │   ├── data/processed/
│   │   ├── models/
│   │   │   ├── saved_models/
│   │   │   └── roberta_fake_review_model/
│   │   ├── scripts/
│   │   ├── src/rag/
│   │   ├── .env
│   │   ├── quick_api_test.py
│   │   └── requirements.txt
|   |   ├── DATA/
|   |   │   ├── raw/fake_reviews_dataset.csv
|   |   │   └── processed/
|   |   ├── complete_presentation_graphs/
│   └── frontend/
│       ├── src/
│       ├── package.json
│       └── vite.config.js
```


## Tech Stack

**Backend:** FastAPI, scikit-learn, PyTorch, Transformers, FAISS, Sentence-BERT, OpenAI GPT-4o-mini  
**Frontend:** React 18, Vite 5  
**Tools:** Google Colab (T4 GPU), NLTK


## Performance Results

| Model | Week | Accuracy | Precision | Recall | F1-Score |
|-------|------|----------|-----------|--------|----------|
| Logistic Regression | 1 | 93.47% | 0.942 | 0.927 | 0.934 |
| Naive Bayes | 1 | 88.97% | 0.895 | 0.883 | 0.889 |
| SVM Linear | 1 | 93.76% | 0.939 | 0.935 | 0.937 |
| **RoBERTa** | **3** | **95.75%** | **0.958** | **0.957** | **0.957** |

**Key Findings:**
- RoBERTa crosses 95% production-ready threshold
- Traditional ML plateaus at ~93% accuracy
- RAG system provides transparent, evidence-based explanations


## Dataset

**Source:** Fake Reviews Dataset (Salminen et al., 2021) - https://osf.io/tyue9/  
**Size:** 40,000 reviews (50% real from Amazon 2018, 50% GPT-2 generated)  
**Categories:** 10 product types (Books, Electronics, Home & Kitchen, etc.)  
