import cv2
import numpy as np
import os
import time
import psycopg2
from tqdm import tqdm
from insightface.app import FaceAnalysis

# DB CONFIG INSTRUCTIONS

# CREATE EXTENSION IF NOT EXISTS vector;

# CREATE TABLE faces (
#     id SERIAL PRIMARY KEY,
#     photo_key TEXT NOT NULL,
#     embedding VECTOR(512) NOT NULL
# );

# CREATE INDEX faces_embedding_idx
# ON faces
# USING hnsw (embedding vector_l2_ops)
# WITH (m = 16, ef_construction = 200);

# PostgreSQL connection setup
DB_PARAMS = {
    "dbname": "recognition_db",
    "user": "postgres",
    "host": "localhost",
    "port": "5432",
}

# Connect to PostgreSQL
conn = psycopg2.connect(**DB_PARAMS)
cursor = conn.cursor()

# Initialize face analysis model
app = FaceAnalysis(name="antelopev2")
app.prepare(ctx_id=-1)  # CPU (-1) or GPU (0)

# Directory containing dataset images
dataset_dir = "/Users/sebasdeldi/Development/SD/human_recognition/test_runner_1"

def store_embedding(image_name, embedding):
    normalized_embedding = embedding / np.linalg.norm(embedding)  # Normalize
    cursor.execute(
        "INSERT INTO embeddings (name, embedding) VALUES (%s, %s)",
        (image_name, normalized_embedding.astype(np.float32).tolist()),  # Ensure float32
    )
    conn.commit()

def process_image(filename):
    """Extract face embeddings from an image and store them."""
    image_path = os.path.join(dataset_dir, filename)

    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Skipping {filename}, failed to load.")
        return

    start_time = time.time()
    faces = app.get(img)
    extraction_time = time.time() - start_time

    for face in faces:
        embedding = face.embedding
        store_embedding(filename, embedding)

    print(f"✅ Processed {filename} with {len(faces)} faces in {extraction_time:.2f} seconds.")

# Process all images in the dataset
total_start_time = time.time()
for filename in tqdm(os.listdir(dataset_dir)):
    process_image(filename)

print(f"✅ Completed in {time.time() - total_start_time:.2f} seconds.")

cursor.close()
conn.close()
