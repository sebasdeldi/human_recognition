import cv2
import numpy as np
import os
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from insightface.app import FaceAnalysis

total_time = time.time()
# ✅ Use SCRFD detector with a high-performance ArcFace model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)  # ✅ Use GPU if available, set to -1 for CPU

# ✅ Image Preprocessing: Resize for faster processing
def resize_image(img, max_size=800):
    """Resize image while maintaining aspect ratio."""
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img

# ✅ Load Reference Image (Angelina)
ref_image_path = "/Users/sebasdeldi/Development/SD/recognition-api/angelina.jpg"
ref_image = cv2.imread(ref_image_path)
if ref_image is None:
    raise ValueError("❌ Reference image not found!")

# ✅ Detect face and extract embedding from reference image
start_time = time.time()
ref_faces = app.get(ref_image)
ref_extraction_time = time.time() - start_time

if not ref_faces:
    raise ValueError("❌ No face detected in the reference image!")

ref_embedding = ref_faces[0].embedding  # ✅ Use unnormalized embedding for accuracy
print(f"✅ Extracted reference face in {ref_extraction_time:.2f} seconds")

# ✅ Define dataset directory and similarity threshold
dataset_dir = "/Users/sebasdeldi/Development/SD/recognition-api/people_dataset"
threshold = 0.5  # Adjusted threshold for better accuracy

# ✅ Parallel Image Processing
matching_images = []

def process_image(filename):
    """Process a single image: detect faces, compute similarity, and check for matches."""
    image_path = os.path.join(dataset_dir, filename)

    # ✅ Skip non-image files
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Skipping {filename}, failed to load.")
        return None

    # img = resize_image(img, max_size=800)  # ✅ Resize for faster detection

    # ✅ Detect faces
    start_time = time.time()
    faces = app.get(img)
    extraction_time = time.time() - start_time

    if not faces:
        return None  # No faces detected

    best_match = None  # Store highest similarity
    for face in faces:
        embedding = face.embedding  # ✅ Use unnormalized embedding
        similarity = np.dot(ref_embedding, embedding) / (np.linalg.norm(ref_embedding) * np.linalg.norm(embedding))  # Cosine similarity

        if similarity > threshold:
            if best_match is None or similarity > best_match[1]:
                best_match = (filename, similarity)

    if best_match:
        print(f"🕒 Processed {filename} in {extraction_time:.2f} seconds (Similarity: {best_match[1]:.2f})")
    else:
        print(f"🕒 Processed {filename} in {extraction_time:.2f} seconds (No match found)")

    return best_match

# ✅ Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust workers based on CPU/GPU power
    results = list(tqdm(executor.map(process_image, os.listdir(dataset_dir)), total=len(os.listdir(dataset_dir))))

# ✅ Collect valid matches
matching_images = [res for res in results if res is not None]

# ✅ Print Results
if matching_images:
    matching_images.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity
    print("\n✅ Matching images:")
    for img, score in matching_images:
        print(f"- {img} (Similarity: {score:.2f})")
    print(f"\n✅ Found {len(matching_images)} matching images.")
else:
    print("❌ No matches found.")

print (f"Total time: {time.time() - total_time:.2f} seconds")