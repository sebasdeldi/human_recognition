import os
import cv2
import requests
import numpy as np
import psycopg2
import time
import modal
from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
MODEL_VOLUME_NAME = "insightface-models"
VOLUME_MOUNT_PATH = "/root/.insightface/models"
DATABASE_URL = os.getenv("DATABASE_URL")
MAX_DOWNLOAD_WORKERS = 5      # For concurrent image downloads
DEFAULT_TIMEOUT = 10          # Seconds for requests timeout
BATCH_SIZE = 5                # Number of records per batch for insertion

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
    "onnxruntime"
)

app = modal.App(
    image=image,
    name="insightface-app",
    secrets=[modal.Secret.from_name("face-extraction-secrets")]
)

# -----------------------------------------------------------------------------
# Global Resources: Requests Session, FaceAnalysis model & Database Connection
# -----------------------------------------------------------------------------
session = requests.Session()
face_analyzer = None
db_conn = None

def get_face_analyzer():
    """
    Lazily initialize and return the FaceAnalysis model using CPU (ctx_id=-1).
    """
    global face_analyzer
    if face_analyzer is None:
        print("Initializing FaceAnalysis model...")
        face_analyzer = FaceAnalysis(name="antelopev2")
        face_analyzer.prepare(ctx_id=-1)
    return face_analyzer

def get_db_connection():
    """
    Lazily initialize and return a persistent PostgreSQL connection.
    """
    global db_conn
    if db_conn is None:
        print("Establishing PostgreSQL connection...")
        db_conn = psycopg2.connect(DATABASE_URL)
    return db_conn

def download_image(url, retries=3):
    """
    Download an image from a URL with retry logic.
    Returns the image content on success.
    """
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"Attempt {attempt+1} failed for URL {url}: {e}")
            time.sleep(1)
    raise Exception(f"Failed to download image from {url} after {retries} attempts.")

def process_image(item, image_content, face_analyzer_instance):
    """
    Decode the image, run face analysis to extract embeddings,
    and return a list of tuples (photo_key, normalized_embedding).
    """
    photo_key = item.get("photo_key")
    decode_start = time.time()
    img_array = np.frombuffer(image_content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    decode_duration = time.time() - decode_start
    if img is None:
        print(f"Failed to decode image {photo_key}")
        return []
    print(f"Image {photo_key} decoded with dimensions: {img.shape} in {decode_duration:.2f} seconds")

    extraction_start = time.time()
    faces = face_analyzer_instance.get(img)
    extraction_duration = time.time() - extraction_start
    print(f"Extracted {len(faces)} faces from {photo_key} in {extraction_duration:.2f} seconds")

    records = []
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
        records.append((photo_key, normalized_embedding))
    return records

# -----------------------------------------------------------------------------
# Modal Function to Process Images
# -----------------------------------------------------------------------------
@app.function(volumes={VOLUME_MOUNT_PATH: model_volume}) # cpu=4.0, memory=3.0
def analyze_and_store_embeddings(payload):
    """
    Processes a batch of images:
      - Uses a globally-initialized FaceAnalysis model.
      - Uses a globally-established PostgreSQL connection.
      - Downloads images concurrently.
      - Processes each image sequentially for face analysis.
      - Stores embeddings in the PostgreSQL table 'faces' via bulk insert,
        batching 5 records at a time.
    
    Payload format:
      {
          "image_list": [
              {"photo_key": "<id>", "image_url": "<url>"},
              ...
          ]
      }
    
    Returns a summary dict including timing information.
    """
    overall_start = time.time()

    # Get global resources
    face_analyzer_instance = get_face_analyzer()
    conn = get_db_connection()
    cursor = conn.cursor()

    # Download images concurrently
    image_list = payload.get("image_list", [])
    download_start = time.time()
    downloaded_images = {}  # Map photo_key -> image content

    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
        future_to_item = {executor.submit(download_image, item.get("image_url")): item for item in image_list}
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            photo_key = item.get("photo_key")
            try:
                downloaded_images[photo_key] = future.result()
                print(f"Downloaded image {photo_key} ({len(downloaded_images[photo_key])} bytes)")
            except Exception as e:
                print(f"Failed to download image {photo_key}: {e}")
    download_end = time.time()
    print(f"Concurrent image downloads completed in {download_end - download_start:.2f} seconds")

    # Process images sequentially for face analysis
    processing_start = time.time()
    insert_records = []
    for item in image_list:
        photo_key = item.get("photo_key")
        image_content = downloaded_images.get(photo_key)
        if image_content is None:
            print(f"Skipping image {photo_key} due to download failure")
            continue

        records = process_image(item, image_content, face_analyzer_instance)
        insert_records.extend(records)
    processing_end = time.time()
    print(f"Sequential face analysis completed in {processing_end - processing_start:.2f} seconds")

    # Bulk insert embeddings into PostgreSQL in batches of 5 records
    insert_start = time.time()
    if insert_records:
        for i in range(0, len(insert_records), BATCH_SIZE):
            batch = insert_records[i:i+BATCH_SIZE]
            cursor.executemany(
                "INSERT INTO faces (photo_key, embedding) VALUES (%s, %s)",
                batch
            )
            conn.commit()
            print(f"Bulk inserted {len(batch)} records in batch starting at index {i}")
    else:
        print("No records to insert")
    insert_end = time.time()
    print(f"Database insertion took {insert_end - insert_start:.2f} seconds")

    cursor.close()
    # DB connection remains open for reuse.
    overall_end = time.time()
    print(f"Overall function execution time: {overall_end - overall_start:.2f} seconds")

    summary = {"message": f"Stored embeddings for {len(insert_records)} faces from {len(image_list)} images."}
    print(summary)
    return summary
