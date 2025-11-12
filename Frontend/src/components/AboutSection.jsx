// Filename: AboutSection.jsx
import React from 'react'
import "../App.css";

export default function AboutSection() {
  return (
    <section 
      id="about" 
      className="my-5 py-4 bg-dark text-light rounded-pill shadow-sm"
    >
      <div className="container">
        <h4 className="fw-bold mb-3">About</h4>
        <p className="text-white">
          We help you uncover the truth behind digital media using advanced{" "}
          <strong>DeepFake Detection and Evidence Validation</strong> tools. Our AI-powered 
          platform analyzes videos to ensure authenticity and protect 
          against misinformation.
        </p>
      </div>
    </section>
  )
}
