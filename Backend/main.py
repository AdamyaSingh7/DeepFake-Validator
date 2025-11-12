#RUN: uvicorn main:app --reload

import os
import shutil
import cv2
from typing import List, Dict, Any

# FastAPI imports
import uvicorn
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware  # ✅ Added for frontend integration

# Deepfake module imports
from inference import (
    initialize_detector,
    run_inference_with_gradcam,
    visualize_heatmap
)

# --- 1. Configuration ---

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

# Creating directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. Initializing FastAPI App ---

app = FastAPI(
    title="Deepfake Detection API",
    description="Upload a video to analyze it for deepfakes."
)

# ✅ --- 2.1 Enable CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later to ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. Mount Static Directory ---
app.mount(
    "/static",
    StaticFiles(directory=OUTPUT_DIR),
    name="static_outputs"
)


# --- 4. Load Model on Startup ---
@app.on_event("startup")
def on_startup():
    """Load all models into memory on server startup."""
    print("--- Server starting up, loading models... ---")
    try:
        initialize_detector()
        print("--- Models loaded successfully. ---")
    except Exception as e:
        print(f"FATAL: Could not load models on startup: {e}")


# --- 5. Health Check Endpoint ---
@app.get("/")
def read_root():
    """A simple endpoint to check if the server is running."""
    return {"status": "Deepfake Detection API is running"}


# --- 6. The Main Analysis Endpoint ---
@app.post("/analyze/")
def analyze_video(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Accepts a video file, saves it, runs inference, and returns
    the verdict, confidence, and visual evidence (heatmaps or reference frames).
    """
    temp_video_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # Step 1: Save the uploaded file
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 2: Run Inference
        print(f"Analyzing video: {temp_video_path}")
        result = run_inference_with_gradcam(temp_video_path)
        print("Analysis complete.")

        # Step 3: Handle inference errors
        if result.get('status') == 'error':
            raise HTTPException(status_code=400, detail=result['message'])

        # Step 4: Process results
        is_fake = result.get('is_fake', False)
        probability = result.get('probability', 0.0)

        verdict = "FAKE" if is_fake else "REAL"
        confidence = probability if is_fake else (1.0 - probability)
        heatmap_urls = []

        # --- Step 5: Generate Heatmaps (if FAKE) ---
        if is_fake:
            print("Deepfake detected. Generating heatmap overlays...")
            faces = result.get('faces', [])
            heatmaps = result.get('heatmaps', [])

            for i in range(len(faces)):
                overlay_image = visualize_heatmap(faces[i], heatmaps[i])
                base_filename = os.path.splitext(file.filename)[0]
                output_filename = f"{base_filename}_heatmap_{i:02d}.jpg"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                cv2.imwrite(output_path, overlay_image)

                url = request.url_for('static_outputs', path=output_filename)
                heatmap_urls.append(str(url))

            print(f"Saved {len(heatmap_urls)} heatmap images.")

        else:
            #For REAL videos Saving original reference face frames
            print("Video classified as REAL — saving reference face frames.")
            faces = result.get('faces', [])
            for i, face in enumerate(faces):
                base_filename = os.path.splitext(file.filename)[0]
                ref_filename = f"{base_filename}_reference_{i:02d}.jpg"
                ref_path = os.path.join(OUTPUT_DIR, ref_filename)
                cv2.imwrite(ref_path, face)
                url = request.url_for('static_outputs', path=ref_filename)
                heatmap_urls.append(str(url))
            print(f"Saved {len(heatmap_urls)} reference face images.")

        # --- Step 6: Return JSON response ---
        return {
            "verdict": verdict,
            "confidence": confidence,
            "heatmap_urls": heatmap_urls,
            "file_type": file.content_type
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    finally:
        # Step 7: Cleanup temp upload
        file.file.close()
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print(f"Cleaned up temp file: {temp_video_path}")


# --- 7. Run the Server ---
if __name__ == "__main__":
    print("Starting Uvicorn server at http://0.0.0.0:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
