// Filename: Footer.jsx
import React from 'react'
import '../App.css'

export default function Footer() {
  return (
    <footer className="bg-dark text-light py-3 mt-auto rounded-4">
      <div className="container d-flex justify-content-between align-items-center">
        <div className="small">&copy; {new Date().getFullYear()} Deepfake Validator</div>
        <div className="small">
          <a
            href="mailto:adamyasingh2516@gmail.com"
            target="_blank"
            rel="noreferrer"
            className="text-light text-decoration-none"
          >
            Contact ↗
          </a>
        </div>
      </div>
    </footer>
  )
}
