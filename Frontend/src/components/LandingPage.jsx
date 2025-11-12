import React from "react";
import "../App.css";

export default function LandingPage({ onCTAClick }) {
  return (
    <div
      className="position-relative text-center text-light overflow-hidden rounded-4 shadow-lg my-5"
      style={{
        background: "rgba(18, 18, 18, 0.8)", // dark glass effect
      }}
    >
      {/* Background Video */}
      <video
        className="position-absolute top-0 start-0 w-100 h-100 object-fit-cover"
        autoPlay
        loop
        muted
        playsInline
      >
        <source src="/background.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>

      {/* Dark overlay for readability */}
      <div className="position-absolute top-0 start-0 w-100 h-100 bg-dark opacity-50"></div>

      {/* Foreground Content */}
      <div className="position-relative p-5 py-7" style={{ minHeight: "75vh" }}>
        <h1 className="display-3 fw-bold mb-4 text-gradient">
          DeepFake Detection & Evidence Validator
        </h1>
        <p className="lead mb-5 fs-4">
          Detect the lie — and get a clear, explainable report showing why the
          media is suspicious.
        </p>
        <button
          className="btn btn-lg btn-primary shadow px-5 py-3"
          onClick={onCTAClick}
        >
          🚀 Upload Media for Analysis
        </button>
      </div>
    </div>
  );
}
