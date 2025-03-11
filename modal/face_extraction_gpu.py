import os
import cv2
import requests
import numpy as np
import psycopg2
import time
import modal

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
MODEL_VOLUME_NAME = "insightface-models"
VOLUME_MOUNT_PATH = "/root/.insightface/models"
TARGET_PATH = f"{VOLUME_MOUNT_PATH}/antelopev2"
DATABASE_URL = os.getenv("DATABASE_URL")

# -----------------------------------------------------------------------------
# Setup Modal Volume and Image
# -----------------------------------------------------------------------------
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME)

image = modal.Image.debian_slim().apt_install("libgl1", "libglib2.0-0").pip_install(
    "torch",
    "insightface==0.7.3",
    "opencv-python-headless",
    "requests",
    "numpy",
    "psycopg2-binary",
    "onnxruntime-gpu"
)

app = modal.App(
    image=image,
    name="insightface-app",
    secrets=[modal.Secret.from_name("face-extraction-secrets")]
)

# -----------------------------------------------------------------------------
# Modal Function to Process Images
# -----------------------------------------------------------------------------
@app.function(volumes={VOLUME_MOUNT_PATH: model_volume}, gpu="H100")
def analyze_and_store_embeddings(payload):
    """
    Processes a batch of images by:
      - Verifying that required model files are present.
      - Initializing the InsightFace model using CUDA.
      - Downloading each image and extracting face embeddings.
      - Normalizing each embedding and inserting them into the PostgreSQL table 'faces'.
    
    Expects payload to be a dict with a key "image_list" that is a list of dictionaries.
    Each dictionary should contain:
       - "photo_key": A unique identifier for the image.
       - "image_url": A URL where the image can be downloaded.
    
    Returns a summary dict including timing information.
    """
    import torch
    from insightface.app import FaceAnalysis

    print("CUDA available:", torch.cuda.is_available())

    overall_start = time.time()

    # --- Verify Model Files ---
    model_verif_start = time.time()
    print("Verifying model files...")
    required_files = ['scrfd_10g_bnkps.onnx', 'glintr100.onnx', 'genderage.onnx']
    for f in required_files:
        model_file_path = os.path.join(TARGET_PATH, f)
        if not os.path.exists(model_file_path):
            error_msg = f"Missing required model file: {f}"
            print(error_msg)
            raise FileNotFoundError(error_msg)
    model_verif_end = time.time()
    print(f"Model file verification took {model_verif_end - model_verif_start:.2f} seconds")

    # --- Initialize Face Analysis Model ---
    face_init_start = time.time()
    print("Initializing FaceAnalysis model...")
    face_analyzer = FaceAnalysis(name="antelopev2", providers=['CUDAExecutionProvider'])
    face_analyzer.prepare(ctx_id=-1)
    face_init_end = time.time()
    print(f"FaceAnalysis model initialization took {face_init_end - face_init_start:.2f} seconds")

    # --- Connect to PostgreSQL ---
    db_conn_start = time.time()
    print("Connecting to PostgreSQL database...")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    db_conn_end = time.time()
    print(f"Database connection established in {db_conn_end - db_conn_start:.2f} seconds")

    inserted_records = 0

    processing_start = time.time()
    image_list = payload.get("image_list", [])
    for item in image_list:
        photo_key = item.get("photo_key")
        image_url = item.get("image_url")
        print(f"\nProcessing image: {photo_key} from {image_url}")

        download_start = time.time()
        response = requests.get(image_url)
        response.raise_for_status()
        download_end = time.time()
        print(f"Downloaded image {photo_key} ({len(response.content)} bytes) in {download_end - download_start:.2f} seconds")

        decode_start = time.time()
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        decode_end = time.time()
        if img is None:
            print(f"Failed to decode image {photo_key}")
            continue
        print(f"Image {photo_key} decoded with dimensions: {img.shape} in {decode_end - decode_start:.2f} seconds")

        extraction_start = time.time()
        faces = face_analyzer.get(img)
        extraction_end = time.time()
        print(f"Extracted {len(faces)} faces from {photo_key} in {extraction_end - extraction_start:.2f} seconds")

        insert_start = time.time()
        for face in faces:
            embedding = face.embedding
            if embedding is None:
                print(f"No embedding found for a face in {photo_key}")
                continue
            norm = np.linalg.norm(embedding)
            if norm == 0:
                print(f"Encountered zero-norm embedding in {photo_key}")
                continue
            normalized_embedding = (embedding / norm).astype(np.float32).tolist()

            cursor.execute(
                "INSERT INTO faces (photo_key, embedding) VALUES (%s, %s)",
                (photo_key, normalized_embedding)
            )
            inserted_records += 1
        conn.commit()
        insert_end = time.time()
        print(f"Inserted embeddings for image {photo_key} in {insert_end - insert_start:.2f} seconds")
    processing_end = time.time()
    print(f"Total processing time for all images: {processing_end - processing_start:.2f} seconds")

    cursor.close()
    conn.close()
    print("Database connection closed.")

    overall_end = time.time()
    print(f"Overall function execution time: {overall_end - overall_start:.2f} seconds")

    summary = {
        "message": f"Stored embeddings for {inserted_records} faces from {len(image_list)} images."
    }
    print(summary)
    return summary
