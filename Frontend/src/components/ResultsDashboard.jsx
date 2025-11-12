import React from "react";
import "../App.css";

export default function ResultsDashboard({ result, previewUrl, fileType = "", onReset }) {
  if (!result) return null;

  const { verdict, confidence: rawConfidence, heatmap_urls } = result;

  const confidencePercent =
    typeof rawConfidence === "number"
      ? rawConfidence <= 1
        ? `${(rawConfidence * 100).toFixed(2)}%`
        : `${rawConfidence.toFixed(2)}%`
      : "N/A";

  const explanation =
    verdict === "FAKE"
      ? "AI detected strong signs of manipulation based on texture, lighting, and facial inconsistencies."
      : "No suspicious patterns were found — this media appears authentic.";

  const isVideo = fileType?.startsWith("video");

  
  const downloadReport = async () => {
    const base64Images = await Promise.all(
      (heatmap_urls || []).map(async (url) => {
        try {
          const res = await fetch(url, { mode: "cors" });
          const blob = await res.blob();
          return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result);
            reader.readAsDataURL(blob);
          });
        } catch {
          return null;
        }
      })
    );

    
    let previewBase64 = "";
    if (previewUrl) {
      try {
        const response = await fetch(previewUrl);
        const blob = await response.blob();
        previewBase64 = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result);
          reader.readAsDataURL(blob);
        });
      } catch (e) {
        console.warn("Could not embed preview:", e);
      }
    }

    const heatmapHTML =
      base64Images.length > 0
        ? base64Images
            .map(
              (src, i) =>
                `<div style="display:inline-block;margin:8px;">
                   <img src="${src}" style="width:150px;border-radius:6px;">
                   <div style="text-align:center;color:#ccc;">Frame ${i + 1}</div>
                 </div>`
            )
            .join("")
        : "<p style='color:#999;'>No heatmaps available (media was classified as authentic).</p>";

    const previewHTML = previewBase64
      ? isVideo
        ? `<video src="${previewBase64}" controls style="width:400px;border-radius:8px;margin-top:15px;"></video>`
        : `<img src="${previewBase64}" style="width:400px;border-radius:8px;margin-top:15px;">`
      : "<p style='color:#999;'>No preview available.</p>";

    const html = `
      <html>
        <head><title>Deepfake Analysis Report</title></head>
        <body style="font-family:Arial,sans-serif;padding:20px;background:#121212;color:white;">
          <h1>Deepfake Analysis Report</h1>
          <p><strong>Verdict:</strong> ${verdict}</p>
          <p><strong>Confidence:</strong> ${confidencePercent}</p>
          <p><strong>Explanation:</strong> ${explanation}</p>
          <h3>Detected Heatmaps</h3>
          ${heatmapHTML}
          <h3 style="margin-top:25px;">Preview</h3>
          ${previewHTML}
        </body>
      </html>`;

    const w = window.open("", "_blank");
    w.document.write(html);
    w.document.close();
    w.print();
  };

  return (
    <section className="my-5 text-light">
      <h3 className="fw-bold mb-3">Result</h3>

      <div className="card shadow-lg border-0 p-4 bg-dark text-light">
        {/* Verdict and controls */}
        <div className="d-flex align-items-center justify-content-between mb-3">
          <div>
            <h4 className="mb-1">
              {verdict === "FAKE" ? (
                <span className="badge bg-danger me-2">Fake</span>
              ) : (
                <span className="badge bg-success me-2">Authentic</span>
              )}
              {verdict}
            </h4>
            <div className="small text-white">Confidence: {confidencePercent}</div>
          </div>

          <div>
            <button className="btn btn-outline-primary me-2" onClick={downloadReport}>
              Download Report
            </button>
            <button className="btn btn-outline-light" onClick={onReset}>
              Analyze another
            </button>
          </div>
        </div>

        {/* Explanation */}
        <div className="mb-4">
          <h6 className="fw-bold">Explanation</h6>
          <p>{explanation}</p>
        </div>

        {/* Heatmaps */}
        <div className="mb-4">
          <h6 className="fw-bold">Detected Heatmaps</h6>
          {heatmap_urls?.length ? (
            <div className="d-flex flex-wrap gap-3 justify-content-center mt-3">
              {heatmap_urls.map((url, idx) => (
                <div
                  key={idx}
                  className="p-2 bg-secondary rounded shadow-sm"
                  style={{
                    border: "2px solid #0d6efd",
                    width: 160,
                  }}
                >
                  <img
                    src={url}
                    alt={`Heatmap ${idx + 1}`}
                    className="img-fluid rounded"
                    style={{ width: "100%", height: "auto", objectFit: "cover" }}
                    onError={(e) => {
                      e.currentTarget.style.display = "none";
                      e.currentTarget.parentNode.innerHTML =
                        "<div class='text-muted small text-center'>Image unavailable</div>";
                    }}
                  />
                  <div className="small text-center text-muted mt-1">Frame {idx + 1}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-muted fst-italic mt-2">
              {verdict === "REAL"
                ? "No heatmaps generated — this video was classified as authentic."
                : "No heatmaps available for display."}
            </div>
          )}
        </div>

        {/* Preview */}
        <div>
          <h6 className="fw-bold">Preview</h6>
          {previewUrl ? (
            <div className="border rounded p-3 bg-secondary shadow-sm" style={{ maxWidth: 640 }}>
              {isVideo ? (
                <video src={previewUrl} controls className="w-100 rounded" />
              ) : (
                <img src={previewUrl} alt="preview" className="img-fluid rounded" />
              )}
              <small className="text-muted d-block mt-2">
                The visual heatmaps above highlight regions used by the AI model for decision-making.
              </small>
            </div>
          ) : (
            <div className="text-muted">No preview available.</div>
          )}
        </div>
      </div>
    </section>
  );
}
