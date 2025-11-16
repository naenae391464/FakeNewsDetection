import { useState } from 'react'
import axios from 'axios'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  const [reviewText, setReviewText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const analyzeReview = async () => {
    if (!reviewText.trim()) {
      setError('Please enter a review to analyze')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.post(`${API_URL}/predict`, {
        text: reviewText
      })
      setResult(response.data)
    } catch (err) {
      setError('Error analyzing review. Make sure the backend is running.')
      console.error('Error:', err)
    } finally {
      setLoading(false)
    }
  }

  const clearForm = () => {
    setReviewText('')
    setResult(null)
    setError(null)
  }

  const exampleReviews = {
    fake: "This is an absolutely amazing product! Perfect in every way! Best purchase ever! Highly highly recommend to everyone! Five stars isn't enough! Outstanding quality! Incredible! Fantastic!",
    real: "Decent product overall. Works as described. Had some minor issues with the setup - the instructions could be clearer. Quality is good for the price point. Shipping was fast. Would recommend if you're on a budget."
  }

  const loadExample = (type) => {
    setReviewText(exampleReviews[type])
    setResult(null)
    setError(null)
  }

  const wordCount = reviewText.split(/\s+/).filter(word => word.length > 0).length

  return (
    <div id="root">
      <div className="container">
        {/* Header */}
        <div className="header">
          <h1>Fake Review Detector</h1>
          <p>AI-powered detection using Machine Learning</p>
          <p className="subtitle">
            Trained on 40,000+ Amazon reviews | 93.47% accuracy
          </p>
        </div>

        {/* Main Card */}
        <div className="card">
          {/* Example Buttons */}
          <div className="button-group">
            <button onClick={() => loadExample('fake')} className="btn-fake">
              Load Fake Example
            </button>
            <button onClick={() => loadExample('real')} className="btn-real">
              Load Real Example
            </button>
            <button onClick={clearForm} className="btn-clear">
              Clear
            </button>
          </div>

          {/* Text Input */}
          <div className="input-group">
            <label>Enter Product Review</label>
            <textarea
              rows="8"
              placeholder="Paste or type a product review here..."
              value={reviewText}
              onChange={(e) => setReviewText(e.target.value)}
            />
            <p className="word-count">{wordCount} words</p>
          </div>

          {/* Analyze Button */}
          <button
            onClick={analyzeReview}
            disabled={loading || !reviewText.trim()}
            className="btn-primary"
          >
            {loading ? (
              <>
                <span className="loading-spinner"></span>
                Analyzing...
              </>
            ) : (
              'Analyze Review'
            )}
          </button>

          {/* Error Message */}
          {error && (
            <div className="error-message">
              <p>Error</p>
              <p>{error}</p>
            </div>
          )}

          {/* Result */}
          {result && (
            <div className={result.is_fake ? 'result-fake' : 'result-real'}>
              <div className="result-header">
                <div className="result-badge">
                  {result.is_fake ? 'WARNING' : 'VERIFIED'}
                </div>
                <div>
                  <h2 className="result-title">
                    {result.is_fake ? 'Likely Fake Review' : 'Likely Genuine Review'}
                  </h2>
                  <p className="result-label">{result.label}</p>
                </div>
              </div>

              {/* Confidence Bar */}
              <div className="confidence-section">
                <div className="confidence-header">
                  <span>Confidence Score</span>
                  <span>{(result.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{ width: `${result.confidence * 100}%` }}
                  ></div>
                </div>
              </div>

              {/* Explanation */}
              <div className="explanation">
                {result.is_fake ? (
                  <>
                    <strong>Why it might be fake:</strong>
                    This review exhibits patterns commonly found in computer-generated or fraudulent reviews, such as excessive superlatives, lack of specific details, or unnatural language patterns.
                  </>
                ) : (
                  <>
                    <strong>Why it seems genuine:</strong>
                    This review shows characteristics of authentic human-written content, including balanced opinions, specific details, and natural language flow.
                  </>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Info Card */}
        <div className="info-card">
          <h3>Model Information</h3>
          <div className="info-grid">
            <div className="info-item">
              <p>Algorithm</p>
              <p>Logistic Regression</p>
            </div>
            <div className="info-item">
              <p>Training Data</p>
              <p>40,432 Reviews</p>
            </div>
            <div className="info-item">
              <p>Accuracy</p>
              <p>93.47%</p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="footer">
          <p>Built with React + FastAPI | CSE 572 Project</p>
        </div>
      </div>
    </div>
  )
}

export default App