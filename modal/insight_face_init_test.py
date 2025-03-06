# Instructions for setting up modals storage or volume:
# modal volume create insightface-models
# modal volume put insightface-models antelopev2

import modal
import os
import cv2
import requests
import numpy as np
from insightface.app import FaceAnalysis

MODEL_VOLUME_NAME = "insightface-models"
VOLUME_MOUNT_PATH = "/root/.insightface/models"  # Direct InsightFace path
TARGET_PATH = f"{VOLUME_MOUNT_PATH}/antelopev2"

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME)
app = modal.App("insightface-app")

image = (
    modal.Image.debian_slim()
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("insightface==0.7.3", "opencv-python-headless", "onnxruntime", "requests", "numpy")
)

@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: model_volume},  # Mount directly to target path
)
def analyze_image(image_url: str):
    # --- Model Verification ---
    print("🔍 Verifying model files...")
    required_files = ['scrfd_10g_bnkps.onnx', 'glintr100.onnx', 'genderage.onnx']
    for f in required_files:
        if not os.path.exists(f"{TARGET_PATH}/{f}"):
            raise FileNotFoundError(f"❌ Missing required model file: {f}")
    print("✅ All model files present")

    # --- Image Processing ---
    print("\n🌐 Downloading image from URL:", image_url)
    response = requests.get(image_url)
    response.raise_for_status()
    print(f"⬇️ Downloaded image ({len(response.content)} bytes)")

    img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("❌ Failed to decode image")
    print(f"🖼️ Image decoded. Dimensions: {img.shape}")

    # --- Face Analysis ---
    print("\n🔍 Initializing FaceAnalysis...")
    face_analyzer = FaceAnalysis(name="antelopev2")
    face_analyzer.prepare(ctx_id=-1)
    print("✅ Model ready")

    faces = face_analyzer.get(img)
    print(f"🔎 Found {len(faces)} faces")
    return f"Found {len(faces)} faces"

if __name__ == "__main__":
    test_url = "https://www.universityofcalifornia.edu/sites/default/files/berkeley_faces.jpg"
    with app.run():
        print("RESULT:", analyze_image.remote(test_url).get())