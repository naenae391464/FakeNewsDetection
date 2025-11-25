import { useState } from 'react'
import axios from 'axios'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  const [reviewText, setReviewText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const getConfidenceLevel = (conf) => {
    if (conf >= 0.85) return 'high'
    if (conf >= 0.65) return 'medium'
    return 'low'
  }

  const getConfidenceLabelText = (conf) => {
    if (conf >= 0.85) return 'High'
    if (conf >= 0.65) return 'Moderate'
    return 'Low'
  }

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

  const wordCount = reviewText.split(/\s+/).filter(word => word.length > 0).length

  return (
    <div id="root">
      <div className="app-container">
        {/* Header */}
        <div className="header">
          <h1>Fake Review Detector</h1>
        </div>

        {/* Main Content - Two Column Layout */}
        <div className={`main-content ${result ? 'split-view' : 'single-view'}`}>
          
          {/* Left Column - Input */}
          <div className="input-column">
            <div className="input-card">
              <div className="input-group">
                <label>Product Review</label>
                <textarea
                  rows={result ? "20" : "12"}
                  placeholder="Paste or type a product review here..."
                  value={reviewText}
                  onChange={(e) => setReviewText(e.target.value)}
                />
                <p className="word-count">{wordCount} words</p>
              </div>

              <button
                onClick={analyzeReview}
                disabled={loading || !reviewText.trim()}
                className="btn-analyze"
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

              {result && (
                <button onClick={clearForm} className="btn-clear-simple">
                  Clear & Start Over
                </button>
              )}

              {error && (
                <div className="error-message">
                  <p>{error}</p>
                </div>
              )}
            </div>
          </div>

          {/* Right Column - Results (only shows when result exists) */}
          {result && (
            <div className="results-column">
              <div className="results-card">
                
                {/* Classification Result */}
                <div className={`result-header ${result.is_fake ? 'fake' : 'real'}`}>
                  <div className="result-badge">
                    {result.is_fake ? 'FAKE' : 'GENUINE'}
                  </div>
                  <h2 className="result-title">
                    {result.is_fake ? 'Likely Fake Review' : 'Likely Genuine Review'}
                  </h2>
                </div>

                {/* Confidence */}
                <div className="confidence-box">
                  <span className="confidence-label">Confidence</span>
                  <div className="confidence-value">
                    <span className={`conf-badge ${getConfidenceLevel(result.confidence)}`}>
                      {getConfidenceLabelText(result.confidence)}
                    </span>
                    <span className="conf-percent">{(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>

                {/* AI Analysis */}
                {result.rag_explanation && (
                  <div className="analysis-section">
                    <h3>Analysis</h3>
                    <p>{result.rag_explanation}</p>
                  </div>
                )}

                {/* Evidence Examples */}
                {result.evidence && result.evidence.similar_examples && (
                  <div className="evidence-section">
                    <h3>{result.is_fake ? 'Similar Fake Patterns' : 'Similar Genuine Reviews'}</h3>
                    <div className="evidence-list">
                      {result.evidence.similar_examples.slice(0, 3).map((ex, idx) => (
                        <div key={idx} className="evidence-item">
                          <div className="evidence-meta">
                            <span className="similarity">{(ex.similarity * 100).toFixed(0)}% match</span>
                            {ex.rating && (
                              <span className="rating">{ex.rating.toFixed(1)} ★</span>
                            )}
                          </div>
                          <p className="evidence-text">"{ex.text.substring(0, 180)}..."</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Pattern Analysis */}
                {result.patterns && (
                  <div className="patterns-section">
                    <h3>Detected Patterns</h3>
                    <div className="pattern-list">
                      
                      {result.patterns.repetitive_words && Object.keys(result.patterns.repetitive_words).length > 0 && (
                        <div className="pattern-item warn">
                          <span className="icon">⚠</span>
                          <div>
                            <strong>Repetitive Words</strong>
                            <p>{Object.entries(result.patterns.repetitive_words).map(([w, c]) => `"${w}" (${c}×)`).join(', ')}</p>
                          </div>
                        </div>
                      )}
                      
                      {result.patterns.generic_phrases && result.patterns.generic_phrases.length > 0 && (
                        <div className="pattern-item warn">
                          <span className="icon">⚠</span>
                          <div>
                            <strong>Generic Phrases</strong>
                            <p>{result.patterns.generic_phrases.join(', ')}</p>
                          </div>
                        </div>
                      )}
                      
                      {result.patterns.has_specifics ? (
                        <div className="pattern-item ok">
                          <span className="icon">✓</span>
                          <div>
                            <strong>Specific Details Found</strong>
                            <p>Contains numbers, prices, or timeframes</p>
                          </div>
                        </div>
                      ) : (
                        <div className="pattern-item warn">
                          <span className="icon">⚠</span>
                          <div>
                            <strong>Lacks Specifics</strong>
                            <p>No concrete details mentioned</p>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Genuine Contrast */}
                {result.is_fake && result.evidence && result.evidence.genuine_contrast && (
                  <div className="contrast-section">
                    <h3>What Genuine Reviews Look Like</h3>
                    <div className="contrast-list">
                      {result.evidence.genuine_contrast.map((ex, idx) => (
                        <div key={idx} className="contrast-item">
                          <span className="contrast-rating">{ex.rating.toFixed(1)} ★</span>
                          <p>"{ex.text}..."</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  )
}

export default App