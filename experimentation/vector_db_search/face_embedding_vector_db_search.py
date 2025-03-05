import psycopg2
import numpy as np
import time

# Database connection
conn = psycopg2.connect(
    dbname="recognition_db",
    user="postgres",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Search parameters
TOP_K = 10000  # Number of top results
THRESHOLD = 0.75  # Adjust as needed

def search_similar_images(embedding):
    """Search for images in the database that match a given face embedding."""
    start_time = time.time()

    # ✅ Correct way to set `hnsw.ef_search`
    cursor.execute("SET hnsw.ef_search = 100;")

    query = """
        SELECT name, (embedding <=> %s::vector) AS distance
        FROM embeddings
        WHERE (embedding <=> %s::vector) < %s
        ORDER BY distance ASC
        LIMIT %s;
    """
    
    # Convert NumPy array to list
    embedding_list = embedding.astype(np.float32).tolist()

    cursor.execute(query, (embedding_list, embedding_list, THRESHOLD, TOP_K))
    results = cursor.fetchall()

    elapsed_time = time.time() - start_time
    print(f"🔍 Search completed in {elapsed_time:.2f} seconds.")
    print(f"✅ Found {len(results)} matching images.")

    return results

# Extract face embeddings from an image
def extract_embedding(image_path):
    """Load an image and extract its face embeddings."""
    from insightface.app import FaceAnalysis
    import cv2

    app = FaceAnalysis(name="antelopev2")
    app.prepare(ctx_id=-1)

    img = cv2.imread(image_path)
    faces = app.get(img)

    if not faces:
        print("❌ No face detected in the input image.")
        return None

    return faces[0].embedding.astype(np.float32)  # Ensure float32

# Example usage
input_image_path = "/Users/sebasdeldi/Development/SD/human_recognition/ronaldo.webp"
embedding = extract_embedding(input_image_path)

if embedding is not None:
    search_results = search_similar_images(embedding)
    for img_name, similarity in search_results:
        print(f"- {img_name} (Distance: {similarity:.4f})")

cursor.close()
conn.close()
