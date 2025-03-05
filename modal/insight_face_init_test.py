# import modal
# import os
# import cv2
# import insightface

# # Modal App
# app = modal.App("insightface-test")

# # Define Image with dependencies
# image = (
#     modal.Image.debian_slim()
#     .apt_install("libgl1", "libglib2.0-0")
#     .pip_install("insightface", "onnxruntime", "opencv-python", "numpy")
# )

# # Attach Model Volume
# VOLUME_NAME = "insightface-models"
# volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

# @app.function(image=image, volumes={"/model_cache": volume})
# def test_insightface():
#     """Loads InsightFace model and processes an image."""
#     print("🚀 Starting test_insightface()")
    
#     model_path = "/antelopev2"
    
#     # Ensure model exists before loading
#     if not os.path.exists(os.path.join(model_path, "glintr100.onnx")):
#         print("❌ Model not found in Modal volume.")
#         return "❌ Model not found. Upload it manually to Modal volume."

#     print("✅ Model found! Loading InsightFace...")

#     # Load the model
#     face_app = insightface.app.FaceAnalysis(name="antelopev2", root=model_path)
#     face_app.prepare(ctx_id=-1)  # Use CPU (-1) or GPU (0)
#     print("✅ InsightFace model loaded!")

#     # Load a test image
#     image_url = "https://www.universityofcalifornia.edu/sites/default/files/berkeley_faces.jpg"
#     print(f"📥 Downloading image from {image_url}...")

#     import urllib.request
#     img_path = "/tmp/test_image.jpg"
#     urllib.request.urlretrieve(image_url, img_path)

#     img = cv2.imread(img_path)
    
#     if img is None:
#         print("❌ Failed to load image.")
#         return "❌ Failed to load image."

#     print("✅ Image loaded successfully!")

#     # Detect faces
#     faces = face_app.get(img)
#     print(f"✅ InsightFace detected {len(faces)} face(s).")
    
#     return f"✅ InsightFace detected {len(faces)} face(s)."


# if __name__ == "__main__":
#     with app.run():
#         result = test_insightface.remote()
#         print(result.get(timeout=60))  # Fetch and print the result


# import modal
# import os
# import cv2
# from insightface.app import FaceAnalysis

# MODEL_VOLUME_NAME = "insightface-models"
# VOLUME_MOUNT_PATH = "/mnt/models"  # Where we mount the volume
# TARGET_PATH = "/root/.insightface/models/antelopev2"  # InsightFace's expected path

# model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME)
# app = modal.App("insightface-app")

# image = (
#     modal.Image.debian_slim()
#     .apt_install("libgl1", "libglib2.0-0")
#     .pip_install("insightface==0.7.3", "opencv-python-headless", "onnxruntime")
# )

# @app.function(
#     image=image,
#     volumes={VOLUME_MOUNT_PATH: model_volume},
# )
# def analyze_image():
#     # Copy models from volume to InsightFace's expected location
#     os.makedirs(TARGET_PATH, exist_ok=True)
    
#     # Check if models exist in volume
#     if not os.path.exists(f"{VOLUME_MOUNT_PATH}/antelopev2"):
#         raise FileNotFoundError("Models not found in volume! Upload them with:\n"
#                               "modal volume put insightface-models ./local/path antelopev2")
    
#     # Copy files (only needs to run once per container)
#     if not os.path.exists(f"{TARGET_PATH}/scrfd_10g_bnkps.onnx"):
#         os.system(f"cp -r {VOLUME_MOUNT_PATH}/antelopev2/* {TARGET_PATH}/")

#     # Initialize model
#     app = FaceAnalysis(name="antelopev2")
#     app.prepare(ctx_id=-1)
    
#     # Test with sample image
#     img = cv2.imread("/test.jpg")
#     faces = app.get(img)
#     return f"Found {len(faces)} faces"

# if __name__ == "__main__":
#     with app.run():
#         print(analyze_image.remote())






# Instructions for setting up modals storage or volume:
# modal volume create insightface-models
# modal volume put insightface-models antelopev2

# import modal
# import os
# import cv2
# import requests
# import numpy as np
# from insightface.app import FaceAnalysis

# MODEL_VOLUME_NAME = "insightface-models"
# VOLUME_MOUNT_PATH = "/mnt/models"
# TARGET_PATH = "/root/.insightface/models/antelopev2"

# model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME)
# app = modal.App("insightface-app")

# image = (
#     modal.Image.debian_slim()
#     .apt_install("libgl1", "libglib2.0-0")
#     .pip_install("insightface==0.7.3", "opencv-python-headless", "onnxruntime", "requests", "numpy")
# )

# @app.function(
#     image=image,
#     volumes={VOLUME_MOUNT_PATH: model_volume},
# )
# def analyze_image(image_url: str):
#     # --- Model Setup Phase ---
#     print("🔄 Starting model setup...")
#     os.makedirs(TARGET_PATH, exist_ok=True)
#     print(f"📂 Target directory created at: {TARGET_PATH}")

#     # Check volume contents
#     volume_model_path = f"{VOLUME_MOUNT_PATH}/antelopev2"
#     if not os.path.exists(volume_model_path):
#         raise FileNotFoundError(f"❌ Models not found in volume at {volume_model_path}")

#     # Copy models if needed
#     if not os.path.exists(f"{TARGET_PATH}/scrfd_10g_bnkps.onnx"):
#         print("🔧 Copying model files from volume to target location...")
#         os.system(f"cp -rv {volume_model_path}/* {TARGET_PATH}/")
#         print(f"✅ Copied {len(os.listdir(volume_model_path))} model files")
#         print("📦 Model files in target:", os.listdir(TARGET_PATH))
#     else:
#         print("📁 Using existing model files in target directory")

#     # --- Image Processing Phase ---
#     print("\n🌐 Downloading image from URL:", image_url)
#     response = requests.get(image_url)
#     response.raise_for_status()
#     print(f"⬇️ Downloaded image ({len(response.content)} bytes)")

#     # Decode image
#     img_array = np.frombuffer(response.content, np.uint8)
#     img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#     if img is None:
#         raise ValueError("❌ Failed to decode image")
#     print(f"🖼️ Image decoded successfully. Dimensions: {img.shape}")

#     # --- Face Analysis Phase ---
#     print("\n🔍 Initializing FaceAnalysis model...")
#     face_analyzer = FaceAnalysis(name="antelopev2")
#     face_analyzer.prepare(ctx_id=-1)
#     print("✅ Model initialized and prepared")

#     print("🧠 Analyzing image for faces...")
#     faces = face_analyzer.get(img)
#     print(f"🔎 Found {len(faces)} faces in the image")

#     # Detailed face information
#     if len(faces) > 0:
#         print("\n📝 Face details:")
#         for i, face in enumerate(faces):
#             print(f"  Face {i+1}:")
#             print(f"    - Bounding Box: {face.bbox}")
#             print(f"    - Detection Score: {face.det_score:.2f}")
#             print(f"    - Embedding Shape: {face.embedding.shape}")

#     return f"✅ Analysis complete: Found {len(faces)} faces"

# if __name__ == "__main__":
#     test_url = "https://www.universityofcalifornia.edu/sites/default/files/berkeley_faces.jpg"
#     print("🚀 Starting face analysis workflow...")
#     with app.run():
#         result = analyze_image.remote(test_url)
#         print("\n⏳ Waiting for results...")
#         final_output = result.get(timeout=60)
#         print("\n🎉 Final Result:", final_output)





import modal
import os
import cv2
import requests
import numpy as np
from insightface.app import FaceAnalysis

MODEL_VOLUME_NAME = "insightface-models2"
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