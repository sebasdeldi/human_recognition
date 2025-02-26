from typing import List, Union, Annotated


import psycopg2.extras  # Ensure extras module is imported
from datetime import datetime

import psycopg2
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from deepface import DeepFace
import numpy as np

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    port="5432",
    database="postgres",
    user="sebasdeldi",
    password=""
)
cursor = conn.cursor()

# Ensure pgvector extension is enabled
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()

# FastAPI App
app = FastAPI()

# Pydantic Model for Image URLs
class ImageURLs(BaseModel):
    urls: List[str]

class SearchPayload(BaseModel):
    target_url: str

# Insert multiple embeddings into the database
@app.post("/direct_search")
def store_embeddings(image_urls: ImageURLs):
    embeddings_data = []

    print("Embedding extraction start:")
    print(datetime.now())
    for url in image_urls.urls:
        try:
            embeddings = DeepFace.represent(url, model_name="Facenet", enforce_detection=False)
            for obj in embeddings:
                embedding = obj['embedding']
                embeddings_data.append((url, np.array(embedding).tolist()))
        except Exception as e:
            return {"error": f"Failed to process {url}: {str(e)}"}
    print("Embedding extraction end:")
    print(datetime.now())


    print("Query time start:")
    print(datetime.now())
    if embeddings_data:
        insert_query = "INSERT INTO identities (img_name, embedding) VALUES %s"
        psycopg2.extras.execute_values(cursor, insert_query, embeddings_data, template="(%s, %s)")
        conn.commit()
    print("Query time end:")
    print(datetime.now())    
    return {"status": "Embeddings stored successfully"}

# Insert multiple embeddings into the database
@app.post("/store_embeddings")
def store_embeddings(image_urls: ImageURLs):
    embeddings_data = []

    print("Embedding extraction start:")
    print(datetime.now())
    for url in image_urls.urls:
        try:
            embeddings = DeepFace.represent(url, model_name="Facenet", enforce_detection=False)
            for obj in embeddings:
                embedding = obj['embedding']
                embeddings_data.append((url, np.array(embedding).tolist()))
        except Exception as e:
            return {"error": f"Failed to process {url}: {str(e)}"}
    print("Embedding extraction end:")
    print(datetime.now())


    print("Query time start:")
    print(datetime.now())
    if embeddings_data:
        insert_query = "INSERT INTO identities (img_name, embedding) VALUES %s"
        psycopg2.extras.execute_values(cursor, insert_query, embeddings_data, template="(%s, %s)")
        conn.commit()
    print("Query time end:")
    print(datetime.now())    
    return {"status": "Embeddings stored successfully"}

# Search for similar images
@app.post("/search")
def search_similar_images(payload: SearchPayload):
    try:
        # Generate embedding for target image
        print("Embedding extraction start:")
        print(datetime.now())
        target_embedding = DeepFace.represent(payload.target_url, model_name="Facenet", enforce_detection=False)[0]['embedding']
        target_embedding = np.array(target_embedding).tolist()
        print("Embedding extraction end:")
        print(datetime.now())

        # Define similarity search query
        threshold = 10  # Adjust this threshold as needed
        print("Query time start:")
        print(datetime.now())
        search_query = """
            SELECT img_name, embedding <-> %s::vector AS distance
            FROM identities
            WHERE embedding <-> %s::vector < %s
            ORDER BY distance
            LIMIT 100;
        """
        cursor.execute(search_query, (target_embedding, target_embedding, threshold))
        results = cursor.fetchall()
        print("Query time end:")
        print(datetime.now())

        return {"matches": [{"img_name": r[0], "distance": r[1]} for r in results]}
    
    except Exception as e:
        return {"error": f"Failed to process search: {str(e)}"}
