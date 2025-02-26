import cv2
import numpy as np
import os
import time
from tqdm import tqdm
from insightface.app import FaceAnalysis

total_time = time.time()

# ✅ Use SCRFD for better face detection
app = FaceAnalysis(name="antelopev2")  # Uses ResNet100 + ArcFace embeddings
app.prepare(ctx_id=-1)  # Set 0 for GPU, -1 for CPU

# Load reference image (Angelina)
ref_image_path = "/Users/sebasdeldi/Development/SD/recognition-api/angelina.jpg"
start_time = time.time()
ref_image = cv2.imread(ref_image_path)

if ref_image is None:
    raise ValueError("❌ Reference image not found!")

ref_faces = app.get(ref_image)
ref_extraction_time = time.time() - start_time

if not ref_faces:
    raise ValueError("❌ No face detected in the reference image!")

ref_embedding = ref_faces[0].embedding  # ✅ Use unnormalized embedding for better accuracy
print(f"✅ Extracted reference face in {ref_extraction_time:.2f} seconds")

# Directory containing dataset images
dataset_dir = "/Users/sebasdeldi/Development/SD/recognition-api/people_dataset"
threshold = 0.47  # ✅ Adjusted threshold for more accurate matching

matching_images = []

# Scan all images in dataset
for filename in tqdm(os.listdir(dataset_dir)):
    image_path = os.path.join(dataset_dir, filename)

    # Skip non-image files
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        continue

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Skipping {filename}, failed to load.")
        continue

    # Measure extraction time
    start_time = time.time()
    faces = app.get(img)
    extraction_time = time.time() - start_time

    print(f"🕒 Processed {filename} in {extraction_time:.2f} seconds")

    for face in faces:
        embedding = face.embedding  # ✅ Use unnormalized embedding
        similarity = np.dot(ref_embedding, embedding) / (np.linalg.norm(ref_embedding) * np.linalg.norm(embedding))  # Cosine similarity

        if similarity > threshold:
            matching_images.append((filename, similarity))
            print(f"🎯 Match found: {filename} (Similarity: {similarity:.2f})")

# Print results
if matching_images:
    print("\n✅ Matching images:")
    for img, score in sorted(matching_images, key=lambda x: x[1], reverse=True):
        print(f"- {img} (Similarity: {score:.2f})")
    print(f"\n✅ Found {len(matching_images)} matching images.")
else:
    print("❌ No matches found.")

print (f"Total time: {time.time() - total_time:.2f} seconds")
