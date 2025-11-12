import React, { useRef } from 'react'
import "../App.css";

export default function UploadSection({ file, setFile, previewUrl, onAnalyze }) {
  const inputRef = useRef()

  const onSelectFile = (e) => {
    const f = e.target.files[0]
    if (!f) return
    if (f.size > 200 * 1024 * 1024) {
      alert('File too large (max 200MB).')
      return
    }
    setFile(f)
  }

  const onDrop = (e) => {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (f) setFile(f)
  }

  return (
    <section className="my-5 text-light">
      <h3 className="fw-bold mb-3">Upload</h3>
      <div
        className="dropzone border border-2 border-dashed rounded-3 p-5 text-center bg-dark mb-4"
        onDrop={onDrop}
        onDragOver={(e) => e.preventDefault()}
        onClick={() => inputRef.current.click()}
        style={{
          cursor: 'pointer',
          boxShadow: '0 0 20px rgba(13, 110, 253, 0.4)',
          transition: 'box-shadow 0.3s ease-in-out',
        }}
      >
        {!file ? (
          <>
            <p className="mb-1 fw-semibold text-white">
              Drag & drop image / video / audio here
            </p>
            <small className="text-white">
              or click to select a file (max 200MB)
            </small>
          </>
        ) : (
          <div className="d-flex align-items-center justify-content-center">
            <div className="me-3">
              {file.type.startsWith('image') && (
                <img
                  src={previewUrl}
                  alt="preview"
                  className="rounded shadow"
                  style={{ width: 120, height: 80, objectFit: 'cover' }}
                />
              )}
              {file.type.startsWith('video') && (
                <video
                  src={previewUrl}
                  className="rounded shadow"
                  style={{ width: 160, maxHeight: 90 }}
                  controls
                />
              )}
              {file.type.startsWith('audio') && (
                <audio
                  src={previewUrl}
                  controls
                  className="shadow-sm"
                  style={{ maxWidth: 200 }}
                />
              )}
            </div>
            <div>
              <div className="fw-bold">{file.name}</div>
              <div className="text-muted small">
                {Math.round(file.size / 1024)} KB
              </div>
            </div>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*,video/*,audio/*"
          onChange={onSelectFile}
          style={{ display: 'none' }}
        />
      </div>

      <div className="d-flex gap-3 justify-content-center">
        <button className="btn btn-success px-4" onClick={onAnalyze}>
          Analyze
        </button>
        <button className="btn btn-outline-danger px-4" onClick={() => setFile(null)}>
          Clear
        </button>
      </div>
    </section>
  )
}
