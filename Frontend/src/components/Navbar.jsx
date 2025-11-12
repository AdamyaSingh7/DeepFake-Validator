import React from 'react'

export default function Navbar() {
  return (
    <nav className="navbar navbar-expand-lg shadow-sm fixed-top bg-dark navbar-dark">
      <div className="container">
        <a className="navbar-brand fw-bold text-primary fs-4" href="#">DeepFake<span className="text-light">Validator</span></a>
        <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#nav" aria-controls="nav" aria-expanded="false">
          <span className="navbar-toggler-icon"></span>
        </button>
        <div className="collapse navbar-collapse" id="nav">
          <ul className="navbar-nav ms-auto">
            <li className="nav-item"><a className="nav-link text-light" href="#about">About</a></li>
            <li className="nav-item"><a className="nav-link text-light" href="https://github.com" target="_blank" rel="noreferrer">Docs</a></li>
          </ul>
        </div>
      </div>
    </nav>
  )
}
