import React, { useState, useEffect } from 'react'
import Navbar from './components/Navbar'
import LandingPage from './components/LandingPage'
import UploadSection from './components/UploadSection'
import ProcessingScreen from './components/ProcessingScreen'
import ResultsDashboard from './components/ResultsDashboard'
import AboutSection from './components/AboutSection'
import Footer from './components/Footer'

export default function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [fileType, setFileType] = useState('') // NEW: keep MIME type for preview selection
  const [status, setStatus] = useState('idle') // 'idle' | 'processing' | 'result'
  const [result, setResult] = useState(null)

  useEffect(() => {
    if (!file) {
      setPreviewUrl('')
      setFileType('')
      return
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    setFileType(file.type || '') // store the MIME type (e.g. "video/mp4", "image/png")
    return () => {
      URL.revokeObjectURL(url)
    }
  }, [file])

  // Updated Analyze: calls backend and sets result
  const handleAnalyze = async () => {
    if (!file) return alert('Please upload a file first')
    setStatus('processing')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('http://127.0.0.1:8000/analyze/', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        // Try to read JSON error detail. Fallback to text.
        let errText = 'Server Error'
        try {
          const errJson = await response.json()
          errText = errJson.detail || JSON.stringify(errJson)
        } catch {
          errText = await response.text()
        }
        alert(`Error: ${errText}`)
        setStatus('idle')
        return
      }

      const data = await response.json()
      // optional: backend could include file_type; but we already have fileType from the upload
      // If backend returns confidence as 0-1, good. If it returns percent already, handle in ResultsDashboard.
      setResult(data)
      setStatus('result')
    } catch (error) {
      console.error('Error analyzing file:', error)
      alert('An error occurred while analyzing the file. Check console for details.')
      setStatus('idle')
    }
  }

  const handleReset = () => {
    setFile(null)
    setResult(null)
    setStatus('idle')
    setPreviewUrl('')
    setFileType('')
  }

  return (
    <div className="d-flex flex-column min-vh-100">
      <Navbar />
      <main className="container my-5 flex-grow-1">
        <LandingPage onCTAClick={() => window.scrollTo({ top: 600, behavior: 'smooth' })} />

        {/* show upload only when idle */}
        {status === 'idle' && (
          <UploadSection
            file={file}
            setFile={setFile}
            previewUrl={previewUrl}
            onAnalyze={handleAnalyze}
          />
        )}

        {status === 'processing' && <ProcessingScreen />}

        {status === 'result' && result && (
          <ResultsDashboard
            result={result}
            previewUrl={previewUrl}
            fileType={fileType}        // NEW: pass fileType so preview works for blob URLs
            onReset={handleReset}
          />
        )}

        <AboutSection />
      </main>
      <Footer />
    </div>
  )
}
