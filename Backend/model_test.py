#!/usr/bin/env python

# ## 1. Imports ##
# All necessary imports from the notebook
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import false
from tqdm.auto import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, auc,
    precision_recall_fscore_support, confusion_matrix, roc_curve
)
import urllib


# ## 2. Environment Setup and Configuration ##

def seed_everything(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Path Configuration ---
# !! IMPORTANT: Update these paths to match your environment !!
DATA_ROOT = "C:\\Users\\adamy\\Downloads\\dataset"  # Path to the dataset
OUTPUT_DIR = "C:\\Users\\adamy\\PycharmProjects\\PythonProject\\PythonProject\\output"  # Path to save outputs and find the model
FACE_CACHE_DIR = "C:\\Users\\adamy\\PycharmProjects\\PythonProject\\PythonProject\\face_cache"  # Path to save/load preprocessed faces
# --- End Path Configuration ---

REAL_DIR = os.path.join(DATA_ROOT, 'Celeb-real')
FAKE_DIR = os.path.join(DATA_ROOT, 'Celeb-synthesis')
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model.pth')

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(FACE_CACHE_DIR, 'test'), exist_ok=True)

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ## 3. Dataset Catalog Creation ##

def create_dataset_df():
    """Creates a DataFrame catalog of the real and fake videos."""
    data = []

    # Load real videos
    real_videos = [f for f in os.listdir(REAL_DIR) if f.endswith('.mp4')]
    for video_name in tqdm(real_videos, desc="Processing real videos catalog"):
        video_path = os.path.join(REAL_DIR, video_name)
        data.append({
            'video_path': video_path,
            'label': 0,  # Real
            'video_name': video_name
        })

    # Load fake videos
    fake_videos = [f for f in os.listdir(FAKE_DIR) if f.endswith('.mp4')]
    for video_name in tqdm(fake_videos, desc="Processing fake videos catalog"):
        video_path = os.path.join(FAKE_DIR, video_name)
        data.append({
            'video_path': video_path,
            'label': 1,  # Fake
            'video_name': video_name
        })

    # Create dataframe and split
    df = pd.DataFrame(data)
    # We split to maintain the same test set as in training
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Dataset statistics:")
    print(f"Total videos: {len(df)}")
    print(
        f"Train set: {len(train_df)} videos (Real: {sum(train_df['label'] == 0)}, Fake: {sum(train_df['label'] == 1)})\n"
        f"Val set: {len(val_df)} videos (Real: {sum(val_df['label'] == 0)}, Fake: {sum(val_df['label'] == 1)})\n"
        f"Test set: {len(test_df)} videos (Real: {sum(test_df['label'] == 0)}, Fake: {sum(test_df['label'] == 1)})")

    return train_df, val_df, test_df


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

        # Download models if not present (requires wget)
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

        # Download models if not present (using urllib)
        download_file(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            prototxt_path
        )
        download_file(
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            caffemodel_path
        )
        self.face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

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

        return best_face, None  # Return None for best_box as it's not used in testing

    def extract_faces_from_video(self, video_path, num_frames=16, device=None):
        if device is not None:
            self.device = device
            self.use_cuda = (device.type == 'cuda')
            try:
                if self.use_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.face_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.face_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except AttributeError:
                self.use_cuda = False

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count == 0:
            cap.release()
            return None

        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        faces = []

        batch_size = 16 if self.use_cuda else 1

        for i in range(0, len(frame_indices), batch_size):
            batch_indices = frame_indices[i:i + batch_size]
            batch_faces = []

            for idx in batch_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if not ret:
                    continue

                if self.use_cuda:
                    torch.cuda.empty_cache()

                face, _ = self.extract_face(frame)
                if face is not None:
                    batch_faces.append(face)

            faces.extend(batch_faces)

        cap.release()

        if len(faces) == 0:
            return None

        while len(faces) < num_frames and len(faces) > 0:
            faces.append(faces[-1])

        if len(faces) >= num_frames:
            return np.array(faces[:num_frames])
        else:
            return None


# ## 5. Video Preprocessing Pipeline ##

def preprocess_videos(df, split, face_extractor, num_frames=16, device=None):
    """Extracts and caches faces from videos in the DataFrame."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        print(f"Using GPU for preprocessing: {torch.cuda.get_device_name(0)}")

    cache_dir = os.path.join(FACE_CACHE_DIR, split)
    os.makedirs(cache_dir, exist_ok=True)

    processed = 0
    skipped = 0
    failed = 0

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Preprocessing {split} videos"):
        output_path = os.path.join(cache_dir, f"{row['video_name']}.npy")

        if os.path.exists(output_path):
            skipped += 1
            continue

        try:
            faces = face_extractor.extract_faces_from_video(row['video_path'], num_frames, device=device)
            if faces is not None:
                np.save(output_path, faces)
                processed += 1
            else:
                failed += 1

            if device.type == 'cuda' and processed % 10 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing {row['video_path']}: {e}")
            failed += 1

    print(f"{split} preprocessing stats: Processed {processed}, Skipped {skipped}, Failed {failed}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()


# ## 6. Video Data Augmentation ##

class VideoAugmentation:
    """Applies augmentations to video frames (only normalization for testing)."""

    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, frames):
        frames = np.array(frames).copy()
        frames = frames.astype(np.float32) / 255.0

        if self.is_train:
            # Apply random brightness and contrast
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)  # Contrast
                beta = random.uniform(-0.2, 0.2)  # Brightness
                frames = np.clip(alpha * frames + beta, 0, 1)

            # Random horizontal flip
            if random.random() > 0.5:
                frames = frames[:, :, ::-1, :].copy()

            # Color jitter
            if random.random() > 0.7:
                for c in range(3):
                    factor = random.uniform(0.8, 1.2)
                    frames[:, :, :, c] = np.clip(frames[:, :, :, c] * factor, 0, 1)

        # Convert to torch tensor
        frames = np.transpose(frames, (0, 3, 1, 2)).copy()  # (N, H, W, C) -> (N, C, H, W)
        return torch.FloatTensor(frames)


# ## 7. PyTorch Dataset for Cached Face Sequences ##

class DeepfakeDataset(Dataset):
    """Loads preprocessed face sequences from .npy files."""

    def __init__(self, df, split, transform=None, num_frames=16):
        self.df = df
        self.split = split
        self.transform = transform
        self.num_frames = num_frames
        self.cache_dir = os.path.join(FACE_CACHE_DIR, split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx, retries=0):
        if retries > 10:
            raise RuntimeError(f"Failed to load valid data after 10 retries for index {idx}")

        try:
            video_name = self.df.iloc[idx]['video_name']
            label = self.df.iloc[idx]['label']
            cache_path = os.path.join(self.cache_dir, f"{video_name}.npy")

            if os.path.exists(cache_path):
                faces = np.load(cache_path)
                if faces.shape[0] < self.num_frames:
                    pad_count = self.num_frames - faces.shape[0]
                    padding = np.repeat(faces[-1:], pad_count, axis=0)
                    faces = np.concatenate([faces, padding], axis=0)
            else:
                # If cached file not available, try another sample
                return self.__getitem__((idx + 1) % len(self), retries + 1)

        except Exception as e:
            # print(f"Warning: Error loading data for index {idx} ({e}). Trying next sample.")
            return self.__getitem__((idx + 1) % len(self), retries + 1)

        if self.transform:
            faces = self.transform(faces)

        return {
            'frames': faces,
            'label': torch.tensor(label, dtype=torch.float32)
        }


# ## 8. 3D Temporal Deepfake Detection Model ##

class EnhancedEfficientFace(nn.Module):
    """The EnhancedEfficientFace model architecture."""

    def __init__(self, num_frames=16):
        super(EnhancedEfficientFace, self).__init__()

        # Load pretrained model
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


# ## 9. Model Testing and Evaluation Routines ##

def test(model, loader, device):
    """Runs the model on the test set and calculates metrics."""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            frames = batch['frames'].to(device)
            labels = batch['label'].unsqueeze(1).to(device)

            outputs = model(frames)
            predictions.append(torch.sigmoid(outputs).detach().cpu().numpy())
            targets.append(labels.cpu().numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    # Compute metrics
    auc_score = roc_auc_score(targets, predictions)
    binary_preds = (predictions > 0.5).astype(int)
    accuracy = accuracy_score(targets, binary_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, binary_preds, average='binary', zero_division=0)
    cm = confusion_matrix(targets, binary_preds)

    # Calculate Equal Error Rate (EER)
    fpr, tpr, thresholds = roc_curve(targets, predictions)
    fnr = 1 - tpr

    eer_threshold_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer_threshold = thresholds[eer_threshold_idx]
    eer = fpr[eer_threshold_idx]

    # Plot ROC curve with EER point marked
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.plot(eer, 1 - eer, 'ro', markersize=8, label=f'EER = {eer:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic with EER')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve_from_test_func.png'))
    plt.close()

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (from test() func)')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0, 1], ['Real', 'Fake'])
    plt.yticks([0, 1], ['Real', 'Fake'])

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_from_test_func.png'))
    plt.close()

    results = {
        'test_auc': auc_score,
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'test_eer': eer,
        'confusion_matrix': cm
    }

    print("\nTest Results:")
    print(f"AUC: {auc_score:.4f}")
    print(f"EER: {eer:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    return results


def visualize_sample_predictions(model, test_loader, device, num_samples=8):
    """Visualize sample predictions from the model."""
    model.eval()
    samples = []

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                images = batch.get('frames', batch.get('images', None))
                labels = batch.get('label', batch.get('labels', None))
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                images, labels = batch[0], batch[1]
            else:
                continue

            if images is None or labels is None:
                continue

            images = images.to(device)
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()

            outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            if outputs.shape[-1] == 1:
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            else:
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

            for i in range(min(len(images), num_samples - len(samples))):
                label_val = labels[i] if isinstance(labels, np.ndarray) else labels[i].item()
                prob_val = probs[i] if isinstance(probs, np.ndarray) else float(probs[i])

                samples.append({
                    'image': images[i].cpu(),
                    'label': int(label_val),
                    'pred': float(prob_val)
                })

            if len(samples) >= num_samples:
                break

    if not samples:
        print("Could not collect any samples for visualization")
        return

    plt.figure(figsize=(15, 12))
    for i, sample in enumerate(samples):
        if i >= num_samples:
            break

        plt.subplot(2, 4, i + 1)

        img = sample['image']

        if len(img.shape) == 4:  # [frames, channels, height, width]
            img = img[0]  # Take first frame

        if len(img.shape) == 3 and img.shape[0] in [1, 3]:  # [channels, height, width]
            img = img.permute(1, 2, 0).numpy()
        elif len(img.shape) == 2:  # [height, width]
            img = img.numpy()

        if len(img.shape) == 3 and img.shape[2] == 1:
            img = img.squeeze(2)

        if img.max() <= 1.0 and img.min() < 0:  # Normalized
            # Assuming standard ImageNet normalization (if used during transform)
            # This is a bit of a guess, as the transform doesn't specify normalization
            # Reversing the simple /255.0 normalization
            img = np.clip(img, 0, 1)

        if len(img.shape) == 3 and img.shape[2] == 3:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap='gray')

        true_label = "Fake" if sample['label'] == 1 else "Real"
        pred_label = "Fake" if sample['pred'] > 0.5 else "Real"
        is_correct = (sample['pred'] > 0.5) == (sample['label'] == 1)
        color = "green" if is_correct else "red"

        plt.title(f"True: {true_label}\nPred: {pred_label} ({sample['pred']:.2f})",
                  color=color, fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_predictions.png'))
    plt.show()
    plt.close()


def plot_confusion_matrix_with_percentages(cm):
    """Plots the confusion matrix with both counts and percentages."""
    cm_norm_row = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6) * 100

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (with Percentages)')
    plt.colorbar()

    classes = ['Real', 'Fake']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]}\n({cm_norm_row[i, j]:.1f}%)",
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_percent.png'))
    plt.close()


def plot_threshold_metrics(model, test_loader, device):
    """Calculates and plots accuracy, precision, recall, and F1 vs. threshold."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Computing threshold metrics"):
            if isinstance(batch, dict):
                frames = batch.get('frames', batch.get('images', None))
                labels = batch.get('label', batch.get('labels', None))
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                frames, labels = batch[0], batch[1]
            else:
                continue
            if frames is None or labels is None: continue

            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1)

            frames = frames.to(device)
            labels = labels.to(device)

            outputs = model(frames)

            if isinstance(outputs, tuple): outputs = outputs[0]

            if outputs.shape[-1] == 1:
                predictions = torch.sigmoid(outputs).cpu().numpy()
            else:
                predictions = torch.softmax(outputs, dim=1)[:, 1:2].cpu().numpy()

            all_predictions.append(predictions)
            all_labels.append(labels.cpu().numpy())

    try:
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
    except ValueError as e:
        print(f"Error concatenating predictions/labels for threshold metrics: {e}")
        return 0.5, None, None  # Return default threshold

    thresholds = np.linspace(0.1, 0.9, 81)
    accuracies, precisions, recalls, f1_scores = [], [], [], []

    for threshold in thresholds:
        binary_preds = (all_predictions > threshold).astype(int)
        accuracy = accuracy_score(all_labels, binary_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, binary_preds, average='binary', zero_division=0
        )
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1 Score', lw=2, color='black')

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal F1 Threshold = {optimal_threshold:.2f}')

    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'threshold_metrics.png'))
    plt.close()

    return optimal_threshold, all_labels, all_predictions


def plot_roc_curve(all_labels, all_predictions):
    """Calculates and plots the ROC curve."""
    if all_labels is None or all_predictions is None:
        print("Skipping ROC curve plot due to missing data.")
        return

    all_labels = all_labels.ravel()
    all_predictions = all_predictions.ravel()

    fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve_from_eval_cell.png'))
    plt.close()


# ## 10. Main Execution ##

if __name__ == "__main__":

    # 1. Setup
    seed_everything()
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # 2. Create Dataset DataFrame
    # We run the full function to ensure the test_df is identical to training
    print("Creating dataset catalogs...")
    _, _, test_df = create_dataset_df()

    # 3. Initialize Face Extractor
    face_extractor = FaceExtractor(face_size=224, device=device)

    # 4. Preprocess Test Videos
    print("\nPreprocessing test videos...")
    preprocess_videos(test_df, 'test', face_extractor, device=device)

    # 5. Create Test DataLoader
    print("Creating test data loader...")
    test_transform = VideoAugmentation(is_train=False)
    test_dataset = DeepfakeDataset(test_df, 'test', transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,  # You can adjust batch size based on your GPU memory
        shuffle=False,
        num_workers=2,  # You can adjust num_workers
        pin_memory=True
    )

    # 6. Initialize and Load Model
    print("\nLoading best model for evaluation...")
    model = EnhancedEfficientFace().to(device)

    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Error: Model file not found at {BEST_MODEL_PATH}")
        print("Please ensure 'best_model.pth' is in the OUTPUT_DIR.")
    else:
        try:
            # Use map_location to load model correctly if trained on GPU and testing on CPU
            checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=false)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print("Model loaded successfully.")

            # 7. Run Test Evaluation
            test_results = test(model, test_loader, device)

            # 8. Export Test Results to CSV
            results_df = pd.DataFrame([test_results])
            results_df.to_csv(os.path.join(OUTPUT_DIR, 'test_results.csv'), index=False)

            # 9. Run Additional Visualizations from Notebook
            try:
                visualize_sample_predictions(model, test_loader, device, num_samples=8)
            except Exception as e:
                print(f"Error in sample visualization: {e}")

            # 10. Plot threshold metrics and get data for final ROC
            optimal_threshold, all_labels, all_predictions = plot_threshold_metrics(model, test_loader, device)
            print(f"Optimal threshold for F1 score: {optimal_threshold:.4f}")

            # 11. Plot Confusion Matrix with Percentages
            cm = np.array(test_results['confusion_matrix'])
            plot_confusion_matrix_with_percentages(cm)

            # 12. Plot final ROC curve
            plot_roc_curve(all_labels, all_predictions)

            print(f"\nTesting completed! All outputs saved to {OUTPUT_DIR}")

        except Exception as e:
            print(f"An error occurred during model loading or testing: {e}")
            # Add traceback for more debugging if needed
            # import traceback
            # traceback.print_exc()