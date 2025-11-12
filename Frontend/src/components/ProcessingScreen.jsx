import React from 'react'
import "../App.css";


export default function ProcessingScreen() {
  return (
    <section className="my-5 text-center">
      <div className="card shadow-lg border-0 p-5">
        <div className="d-flex flex-column align-items-center">
          <div
            className="spinner-border text-primary mb-3"
            role="status"
            aria-hidden="true"
            style={{ width: '3rem', height: '3rem' }}
          ></div>
          <h5 className="fw-bold">Analyzing your media...</h5>
          <p className="text-muted">This may take a few seconds depending on file length.</p>
        </div>
      </div>
    </section>
  )
}
