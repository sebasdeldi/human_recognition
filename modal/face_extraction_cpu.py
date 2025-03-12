import os
import cv2
import requests
import numpy as np
import psycopg2
import time
import modal
import io  # For StringIO
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
BATCH_SIZE = 5                # Insert in batches of 5 records

# -----------------------------------------------------------------------------
# Setup Modal Volume and Image
# -----------------------------------------------------------------------------
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME)

image = modal.Image.debian_slim().apt_install("libgl1", "libglib2.0-0").pip_install(
    "insightface==0.7.3",
    "opencv-python-headless",
    "requests",
    "numpy",
    "psycopg2-binary",
    "onnxruntime"
)

app = modal.App(
    image=image,
    name="FaceExtraction",
    secrets=[modal.Secret.from_name("face-extraction-secrets")]
)

# -----------------------------------------------------------------------------
# Class with Lifecycle Hooks and COPY-based Insertion
# -----------------------------------------------------------------------------
@app.cls(volumes={VOLUME_MOUNT_PATH: model_volume})  # cpu=4.0, memory=3.0
class FaceProcessor:
    @modal.enter()
    def initialize(self):
        """Container entry handler: initialize resources once per container startup."""
        print("Container startup: Initializing global resources...")
        self.session = requests.Session()
        print("Initializing FaceAnalysis model...")
        self.face_analyzer = FaceAnalysis(name="antelopev2")
        self.face_analyzer.prepare(ctx_id=-1)
        print("Establishing PostgreSQL connection...")
        self.db_conn = psycopg2.connect(DATABASE_URL)

    @modal.exit()
    def cleanup(self):
        """Container exit handler: clean up resources."""
        print("Container shutdown: Cleaning up resources...")
        if self.db_conn:
            self.db_conn.close()

    def download_image(self, url, retries=3):
        """
        Download an image from a URL with retry logic.
        Returns the image content on success.
        """
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=DEFAULT_TIMEOUT)
                response.raise_for_status()
                return response.content
            except Exception as e:
                print(f"Attempt {attempt+1} failed for URL {url}: {e}")
                time.sleep(1)
        raise Exception(f"Failed to download image from {url} after {retries} attempts.")

    def process_image(self, item, image_content):
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
        faces = self.face_analyzer.get(img)
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
            # Normalize embedding and convert to list of floats.
            normalized_embedding = (embedding / norm).astype(np.float32).tolist()
            records.append((photo_key, normalized_embedding))
        return records

    def copy_insert_embeddings(self, records):
        """
        Optimized bulk insertion using PostgreSQL COPY.
        Converts records to a tab-separated format and uses copy_from.
        The embedding list is converted to a string.
        """
        if not records:
            return
        buffer = io.StringIO()
        for photo_key, embedding in records:
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            buffer.write(f"{photo_key}\t{embedding_str}\n")
        buffer.seek(0)
        cursor = self.db_conn.cursor()
        cursor.copy_from(buffer, 'faces', sep='\t', columns=('photo_key', 'embedding'))
        self.db_conn.commit()
        cursor.close()

    @modal.method()
    def analyze_and_store_embeddings(self, payload):
        """
        Processes a batch of images:
          - Downloads images concurrently.
          - Processes each image sequentially for face analysis.
          - Stores embeddings in the PostgreSQL table 'faces' using the COPY command in batches of 5.
        
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
        image_list = payload.get("image_list", [])
        download_start = time.time()
        downloaded_images = {}

        # Download images concurrently
        with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
            future_to_item = {executor.submit(self.download_image, item.get("image_url")): item for item in image_list}
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
            records = self.process_image(item, image_content)
            insert_records.extend(records)
        processing_end = time.time()
        print(f"Sequential face analysis completed in {processing_end - processing_start:.2f} seconds")

        # Insert embeddings in batches of BATCH_SIZE records using COPY
        insert_start = time.time()
        if insert_records:
            total_records = len(insert_records)
            for i in range(0, total_records, BATCH_SIZE):
                batch = insert_records[i:i+BATCH_SIZE]
                self.copy_insert_embeddings(batch)
                print(f"Bulk inserted batch of {len(batch)} records using COPY.")
        else:
            print("No records to insert")
        insert_end = time.time()
        print(f"Database insertion took {insert_end - insert_start:.2f} seconds")

        overall_end = time.time()
        print(f"Overall function execution time: {overall_end - overall_start:.2f} seconds")
        summary = {"message": f"Stored embeddings for {len(insert_records)} faces from {len(image_list)} images."}
        print(summary)
        return summary
