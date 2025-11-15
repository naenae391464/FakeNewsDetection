import { useState } from 'react'
import axios from 'axios'

const API_URL = 'http://localhost:8000/api'

function App() {
  const [reviewText, setReviewText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [recommendations, setRecommendations] = useState([])

  const analyzeReview = async () => {
    setLoading(true)
    try {
      const response = await axios.post(`${API_URL}/detect`, {
        text: reviewText,
        category: 'Electronics'
      })
      setResult(response.data)

      // If fake, get recommendations
      if (response.data.is_fake) {
        const recResponse = await axios.post(`${API_URL}/recommend`, {
          text: reviewText,
          category: 'Electronics'
        })
        setRecommendations(recResponse.data.alternatives)
      }
    } catch (error) {
      console.error('Error:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Fake Review Detector</h1>
        
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <textarea
            className="w-full border rounded p-3 mb-4"
            rows="6"
            placeholder="Paste a product review here..."
            value={reviewText}
            onChange={(e) => setReviewText(e.target.value)}
          />
          
          <button
            onClick={analyzeReview}
            disabled={loading || !reviewText}
            className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700 disabled:bg-gray-400"
          >
            {loading ? 'Analyzing...' : 'Analyze Review'}
          </button>
        </div>

        {result && (
          <div className={`p-6 rounded-lg ${result.is_fake ? 'bg-red-50' : 'bg-green-50'}`}>
            <h2 className="text-xl font-bold mb-2">
              {result.is_fake ? '⚠️ Likely Fake' : '✅ Likely Genuine'}
            </h2>
            <p>Confidence: {(result.confidence * 100).toFixed(1)}%</p>
            <p className="text-sm text-gray-600">
              Processing time: {result.processing_time_ms}ms
            </p>
          </div>
        )}

        {recommendations.length > 0 && (
          <div className="mt-6 bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-bold mb-4">Recommended Alternatives</h3>
            {recommendations.map((rec, idx) => (
              <div key={idx} className="border-b pb-3 mb-3">
                <p className="text-sm">{rec.text}</p>
                <span className="text-xs text-gray-500">
                  Rating: {rec.rating} ⭐ | Similarity: {(rec.similarity_score * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default App