

# ## 1. Imports ##
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import urllib.request
import sys
from typing import Optional, Dict, Any
import copy
import matplotlib.pyplot as plt

# ## 2. Configuration ##

# !! IMPORTANT: Update these paths to match your environment !!
# Path to save/find the OpenCV face detector models
OUTPUT_DIR = "C:\\Users\\adamy\\PycharmProjects\\PythonProject\\PythonProject\\output"  # Path to save outputs and find the model
# Path to find the trained deepfake model weights
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model.pth')
# --- End Path Configuration ---

# Create output directory if it doesn't exist (for face detector models)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ## 3. Global Variables for Models ##
# These will be initialized once by initialize_detector()
g_model: Optional[nn.Module] = None
g_face_extractor: Optional['FaceExtractor'] = None
g_transform: Optional['VideoAugmentation'] = None
g_device: Optional[torch.device] = None
g_hook_activations: Optional[torch.Tensor] = None
g_hook_gradients: Optional[torch.Tensor] = None


# ## 4. Face Extraction with OpenCV DNN ##

class FaceExtractor:
    """Uses OpenCV's DNN face detector with GPU support to extract faces."""

    def __init__(self, face_size=224, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.use_cuda = (device.type == 'cuda')

        prototxt_path = os.path.join(OUTPUT_DIR, 'deploy.prototxt')
        caffemodel_path = os.path.join(OUTPUT_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')

        # Download models if not present
        def download_file(url, out_path):
            """Downloads a file from a URL to a specified path."""
            if not os.path.exists(out_path):
                print(f"Downloading {os.path.basename(out_path)}...")
                try:
                    with urllib.request.urlopen(url) as response, open(out_path, 'wb') as out_file:
                        data = response.read() # Read all data from URL
                        out_file.write(data)   # Write data to file
                    print("Download complete.")
                except Exception as e:
                    print(f"Error downloading {url}: {e}")
                    print(f"Please download the file manually and place it in the '{OUTPUT_DIR}' directory.")
                    raise

        download_file(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            prototxt_path
        )
        download_file(
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            caffemodel_path
        )

        try:
            self.face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        except cv2.error as e:
            print(f"Error loading Caffe model: {e}")
            print("Please ensure the .prototxt and .caffemodel files are correctly downloaded in the OUTPUT_DIR.")
            raise

        if self.use_cuda:
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.face_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.face_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    print("Using CUDA for face detection")
                else:
                    print("OpenCV not built with CUDA support, using CPU for face detection")
            except AttributeError:
                print("OpenCV CUDA support not available or version mismatch, using CPU for face detection")
                self.use_cuda = False

        self.face_size = face_size

    def extract_face(self, frame, confidence_threshold=0.5):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()

        best_face = None
        best_confidence = confidence_threshold

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > best_confidence:
                best_confidence = confidence
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                width, height = x2 - x1, y2 - y1
                x1 = max(0, x1 - int(width * 0.1))
                y1 = max(0, y1 - int(height * 0.1))
                x2 = min(w, x2 + int(width * 0.1))
                y2 = min(h, y2 + int(height * 0.1))

                if x2 - x1 > 0 and y2 - y1 > 0:
                    face = frame[y1:y2, x1:x2]
                    try:
                        face = cv2.resize(face, (self.face_size, self.face_size))
                        best_face = face
                    except Exception as e:
                        continue

        return best_face

    def extract_faces_from_video(self, video_path, num_frames=16):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count == 0:
            print(f"Error: Could not read video file {video_path}")
            cap.release()
            return None

        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        faces = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                continue

            face = self.extract_face(frame)
            if face is not None:
                faces.append(face)

        cap.release()

        if len(faces) == 0:
            return None # No faces detected

        # Pad the sequence if fewer than num_frames faces were found
        while len(faces) < num_frames and len(faces) > 0:
            faces.append(faces[-1]) # Duplicate the last detected face

        return np.array(faces[:num_frames])


# ## 5. Video Data Augmentation (Transform) ##

class VideoAugmentation:
    """Applies augmentations to video frames (only normalization for inference)."""

    def __init__(self, is_train=False): # Set is_train=False for inference
        self.is_train = is_train

    def __call__(self, frames):
        frames = np.array(frames).copy()
        frames = frames.astype(np.float32) / 255.0
        frames = np.transpose(frames, (0, 3, 1, 2)).copy()  # (N, H, W, C) -> (N, C, H, W)
        return torch.FloatTensor(frames)


# ## 6. 3D Temporal Deepfake Detection Model ##

class EnhancedEfficientFace(nn.Module):
    """The EnhancedEfficientFace model architecture."""

    def __init__(self, num_frames=16):
        super(EnhancedEfficientFace, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(1280, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        x = x.view(batch_size * num_frames, c, h, w)
        features = self.backbone(x)
        features = features.view(batch_size, num_frames, -1)
        features = features.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        temporal_features = self.temporal_conv(features)
        attention_weights = self.attention(temporal_features)
        attended_features = temporal_features * attention_weights
        pooled_features = torch.mean(attended_features, dim=(2, 3, 4))
        output = self.classifier(pooled_features)
        return output


def forward_hook(module, inp, out):
    """Saves the forward-pass activations."""
    global g_hook_activations
    # out is the activation tensor. detach() creates a new tensor
    # that is not part of the computation graph.
    g_hook_activations = out.detach()

def backward_hook(module, grad_in, grad_out):
    """Saves the backward-pass gradients."""
    global g_hook_gradients
    # grad_out is a tuple, we want the first element
    g_hook_gradients = grad_out[0].detach()

# ## 7. Public Functions ##

def initialize_detector():
    """
    Initializes and loads all models into global variables.
    Call this function ONCE on server startup.
    """
    global g_model, g_face_extractor, g_transform, g_device

    print("--- Initializing Deepfake Detector ---")

    # --- 1. Set Device ---
    g_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {g_device}")
    if g_device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # --- 2. Check for Model File ---
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"FATAL ERROR: Model file not found at {BEST_MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found at {BEST_MODEL_PATH}")

    # --- 3. Load Components ---
    try:
        print("Loading face extractor...")
        g_face_extractor = FaceExtractor(face_size=224, device=g_device)

        g_transform = VideoAugmentation(is_train=False)

        print("Loading deepfake detection model...")
        g_model = EnhancedEfficientFace().to(g_device)

        # Load the checkpoint
        checkpoint = torch.load(BEST_MODEL_PATH, map_location=g_device, weights_only=False)

        # Load the model's state dictionary from the checkpoint
        g_model.load_state_dict(checkpoint['model_state_dict'])
        g_model.eval()
        print("--- Detector initialization complete ---")

    except Exception as e:
        print(f"An error occurred during model initialization: {e}")
        # Clean up globals on failure
        g_model = None
        g_face_extractor = None
        g_transform = None
        g_device = None
        raise e


def run_deepfake_inference(video_path: str) -> Dict[str, Any]:
    """
    Runs the full inference pipeline on a single video file.

    Assumes initialize_detector() has already been called.

    Args:
        video_path: The file path to the video to be analyzed.

    Returns:
        A dictionary containing the prediction results.
        Example success:
            {'status': 'success', 'is_fake': True, 'probability': 0.95}
        Example failure:
            {'status': 'error', 'message': 'No faces detected in video.'}
            {'status': 'error', 'message': 'Detector not initialized.'}
    """
    global g_model, g_face_extractor, g_transform, g_device

    # --- 1. Check if models are loaded ---
    if not all([g_model, g_face_extractor, g_transform, g_device]):
        print("Error: Detector models are not initialized. Please call initialize_detector() first.")
        return {'status': 'error', 'message': 'Detector not initialized.'}
    
    # --- 2. Check video file ---
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return {'status': 'error', 'message': f'Video file not found at {video_path}'}

    try:
        # --- 3. Extract faces ---
        faces = g_face_extractor.extract_faces_from_video(video_path, num_frames=16)

        if faces is None:
            print(f"No faces detected in video: {video_path}")
            return {'status': 'error', 'message': 'No faces detected in video.'}

        # --- 4. Apply transform (normalization) ---
        faces_tensor = g_transform(faces)

        # --- 5. Add batch dimension and send to device ---
        # Shape becomes: [1, num_frames, channels, height, width]
        faces_tensor = faces_tensor.unsqueeze(0).to(g_device)

        # --- 6. Run inference ---
        with torch.no_grad():
            output = g_model(faces_tensor)
            # Apply sigmoid to get probability
            probability = torch.sigmoid(output).item()

        is_fake = probability > 0.5
        
        # Return structured result
        return {
            'status': 'success',
            'is_fake': is_fake,
            'probability': probability
        }

    except Exception as e:
        print(f"An error occurred during inference for {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'message': f'An internal error occurred: {e}'}
    
def run_inference_with_gradcam(video_path: str) -> Dict[str, Any]:
    """
    Runs inference and generates Grad-CAM heatmaps for each frame.

    Assumes initialize_detector() has already been called.

    Returns:
        A dictionary containing the prediction and a numpy array of heatmaps.
        Example success:
            {'status': 'success', 'is_fake': True, 'probability': 0.95,
             'faces': (numpy.array, shape (16, 224, 224, 3)),
             'heatmaps': (numpy.array, shape (16, 224, 224))}
        Example failure:
            {'status': 'error', 'message': '...'}
    """
    global g_model, g_face_extractor, g_transform, g_device
    global g_hook_activations, g_hook_gradients

    # --- 1. Check if models are loaded ---
    if not all([g_model, g_face_extractor, g_transform, g_device]):
        print("Error: Detector models are not initialized.")
        return {'status': 'error', 'message': 'Detector not initialized.'}
    
    # --- 2. Check video file ---
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return {'status': 'error', 'message': f'Video file not found at {video_path}'}

    try:
        # --- 3. Extract faces ---
        # We need to keep the original faces for visualization
        original_faces = g_face_extractor.extract_faces_from_video(video_path, num_frames=16)

        if original_faces is None:
            print(f"No faces detected in video: {video_path}")
            return {'status': 'error', 'message': 'No faces detected in video.'}
        
        # Make a copy for transformation
        faces_to_transform = copy.deepcopy(original_faces)

        # --- 4. Apply transform (normalization) ---
        faces_tensor = g_transform(faces_to_transform)

        # --- 5. Add batch dimension and send to device ---
        faces_tensor = faces_tensor.unsqueeze(0).to(g_device)
        
        # --- 6. Register Hooks ---
        # The target layer is the last conv block of the EfficientNet backbone
        target_layer = g_model.backbone.features[8]
        
        # Register the hooks
        f_hook = target_layer.register_forward_hook(forward_hook)
        b_hook = target_layer.register_full_backward_hook(backward_hook)
        
        # Clear any old gradients
        g_model.zero_grad()

        # --- 7. Run Forward Pass ---
        # The forward pass will trigger `farward_hook`
        output = g_model(faces_tensor)
        probability = torch.sigmoid(output).item()
        is_fake = probability > 0.5

        # --- 8. Run Backward Pass ---
        # Backpropagate from the final raw score (the logit)
        # This will trigger `backward_hook`
        output.backward()

        # --- 9. Remove Hooks ---
        # It's crucial to remove hooks so they don't stay attached
        f_hook.remove()
        b_hook.remove()

        # --- 10. Check if hooks captured data ---
        if g_hook_gradients is None or g_hook_activations is None:
            return {'status': 'error', 'message': 'Failed to get grads/activations from hooks.'}
        
        # g_hook_activations shape is (B*N, C, H, W) -> (16, 1280, 7, 7)
        # g_hook_gradients shape is (B*N, C, H, W) -> (16, 1280, 7, 7)
        
        # --- 11. Compute CAMs ---
        # Pool gradients to get neuron importance weights
        pooled_gradients = torch.mean(g_hook_gradients, dim=[2, 3]) # Shape: (16, 1280)

        heatmaps_list = []
        for i in range(g_hook_activations.shape[0]): # Loop through all 16 frames
            frame_activations = g_hook_activations[i] # (1280, 7, 7)
            frame_grad_weights = pooled_gradients[i] # (1280)

            # Weighted sum of activations
            heatmap = torch.zeros((7, 7), device=g_device)
            for k in range(frame_grad_weights.shape[0]): # 1280 channels
                heatmap += frame_grad_weights[k] * frame_activations[k]
            
            heatmap = torch.relu(heatmap) # Apply ReLU
            
            # Normalize for visualization
            if heatmap.max() > 0:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
            # Upscale and convert to numpy
            heatmap_np = heatmap.cpu().numpy()
            heatmap_resized = cv2.resize(heatmap_np, (224, 224))
            heatmaps_list.append(heatmap_resized)
            
        # Clear globals
        g_hook_activations = None
        g_hook_gradients = None

        return {
            'status': 'success',
            'is_fake': is_fake,
            'probability': probability,
            'faces': original_faces,  # The (16, 224, 224, 3) np array
            'heatmaps': np.array(heatmaps_list) # The (16, 224, 224) np array
        }

    except Exception as e:
        print(f"An error occurred during Grad-CAM inference for {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'message': f'An internal error occurred: {e}'}

def visualize_heatmap(face_img: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    Overlays a heatmap onto a face image.

    Args:
        face_img: The original face (H, W, 3) in BGR format (from OpenCV)
        heatmap: The 2D heatmap (H, W) normalized 0-1

    Returns:
        A (H, W, 3) numpy array (BGR) of the face with the heatmap overlay.
    """
    # Use a colormap to color the heatmap
    # 'jet' is a common choice (blue-to-red)
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3] # Get (R, G, B)
    
    # Convert RGB (from matplotlib) to BGR (for OpenCV)
    heatmap_bgr = cv2.cvtColor(heatmap_colored.astype(np.float32), cv2.COLOR_RGB2BGR)

    # Ensure face image is 8-bit for blending
    if face_img.dtype != np.uint8:
         face_img = face_img.astype(np.uint8)

    # Convert heatmap to 8-bit
    heatmap_8bit = (heatmap_bgr * 255).astype(np.uint8)

    # Blend the images
    # 0.5*face + 0.5*heatmap
    overlay = cv2.addWeighted(face_img, 0.5, heatmap_8bit, 0.5, 0)
    
    return overlay

# ## 8. Example Usage (for testing) ##

if __name__ == "__main__":
    """
    This block demonstrates how to use the module.
    It will only run when you execute this file directly:
    python deepfake_detector.py
    """
    
    # --- 1. Initialize the detector (once) ---
    try:
        initialize_detector()
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        sys.exit(1)

    # --- 2. Create a dummy video file for testing ---
    # You should replace this with a real video path
    TEST_VIDEO_PATH = "C:\\Users\\aryas\\Downloads\\vf5.mp4"
    
    if not os.path.exists(TEST_VIDEO_PATH):
        print(f"Warning: Test video '{TEST_VIDEO_PATH}' not found.")
        sys.exit(1)
        # print("Creating a dummy black video file for testing...")
        # try:
        #     # Create a 5-second, 320x240, black video
        #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        #     out = cv2.VideoWriter(TEST_VIDEO_PATH, fourcc, 20.0, (320, 240))
        #     frame = np.zeros((240, 320, 3), dtype=np.uint8)
        #     for _ in range(100): # 5 seconds at 20 fps
        #         out.write(frame)
        #     out.release()
        #     print(f"Dummy video '{TEST_VIDEO_PATH}' created.")
        #     print("Note: The model will likely fail or return 'REAL' for a black video.")
        # except Exception as e:
        #     print(f"Could not create dummy video: {e}")
        #     print("Please provide a real video path in TEST_VIDEO_PATH to test.")
        #     sys.exit(1)
            
    # --- 3. Run inference on the test video ---
    print(f"\n--- Running inference on {TEST_VIDEO_PATH} ---")
    result = run_inference_with_gradcam(TEST_VIDEO_PATH)

    print("\n--- Inference Result (Dictionary) ---")
    print(result)

    # --- 4. Pretty-print the result ---
    print("\n--- Formatted Result ---")
    if result['status'] == 'success':
        label = "FAKE" if result['is_fake'] else "REAL"
        prob = result['probability'] if result['is_fake'] else (1.0 - result['probability'])
        print(f"Prediction: {label}")
        print(f"Confidence: {prob:.2%}")

        # --- THIS IS THE NEW VISUALIZATION PART ---
        print(f"Saving {len(result['heatmaps'])} heatmap overlays to: {OUTPUT_DIR}")
        
        faces = result['faces']
        heatmaps = result['heatmaps']

        for i in range(len(faces)):
            face_img = faces[i]
            heatmap = heatmaps[i]
            
            # Create the overlay
            overlay_image = visualize_heatmap(face_img, heatmap)
            
            # Create a side-by-side comparison
            heatmap_colored = (plt.cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)
            heatmap_colored_bgr = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)
            
            comparison = np.hstack((face_img, heatmap_colored_bgr, overlay_image))
            
            # Save the file
            filename = f"frame_{i:02d}_cam.jpg"
            out_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(out_path, comparison)
        
        print("--- Visualization complete ---")

    else:
        print(f"Error: {result['message']}")