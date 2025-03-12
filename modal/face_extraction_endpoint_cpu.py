import os
import cv2
import numpy as np
import time
import modal
from insightface.app import FaceAnalysis
from fastapi import Request, HTTPException

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
MODEL_VOLUME_NAME = "insightface-models"
VOLUME_MOUNT_PATH = "/root/.insightface/models"

# -----------------------------------------------------------------------------
# Setup Modal Volume and Image
# -----------------------------------------------------------------------------
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME)

image = modal.Image.debian_slim().apt_install("libgl1", "libglib2.0-0").pip_install(
    "insightface==0.7.3",
    "opencv-python-headless",
    "numpy",
    "onnxruntime",
    "fastapi",
    "python-multipart"
)

app = modal.App(
    image=image,
    name="FaceExtractionApi",
    secrets=[modal.Secret.from_name("face-extraction-secrets")]
)

@app.cls(volumes={VOLUME_MOUNT_PATH: model_volume})  # cpu=4.0, memory=3.0
class FaceProcessorEndpoint:
    @modal.enter()
    def initialize(self):
        """Container entry handler: initialize resources once per container startup."""
        print("Container startup: Initializing global resources...")
        print("Initializing FaceAnalysis model...")
        self.face_analyzer = FaceAnalysis(name="antelopev2")
        self.face_analyzer.prepare(ctx_id=-1)

    @modal.fastapi_endpoint(method="POST")
    async def extrac_embedding(self, request: Request):
        """
        Endpoint that accepts an image as multipart/form-data.
          - Validates the AuthorizationToken header against an environment variable.
          - If more than one face is detected, responds with HTTP 422:
              type: MULTIFACE, message: "Photos with only one face allowed"
          - If no face is detected, responds with HTTP 422:
              type: NOFACE, message: "No faces detected in the photo"
          - Otherwise, returns a JSON with the normalized embedding vector.
        """
        # Validate AuthorizationToken header
        auth_token = request.headers.get("AuthorizationToken")
        valid_token = os.environ.get("AUTHORIZATION_TOKEN")
        if not auth_token or auth_token != valid_token:
            raise HTTPException(status_code=401, detail={"message": "Unauthorized"})

        # Parse the form-data from the request
        form = await request.form()
        if "file" not in form:
            raise HTTPException(status_code=400, detail={"message": "Missing file field"})
        
        # Read the uploaded file (expects the field name "file")
        file = form["file"]
        image_content = await file.read()

        # Decode the image
        decode_start = time.time()
        img_array = np.frombuffer(image_content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        decode_duration = time.time() - decode_start
        if img is None:
            raise HTTPException(
                status_code=422,
                detail={"type": "INVALID_IMAGE", "message": "Failed to decode image"}
            )
        print(f"Image decoded with dimensions: {img.shape} in {decode_duration:.2f} seconds")
        
        # Extract faces from the image
        extraction_start = time.time()
        faces = self.face_analyzer.get(img)
        extraction_duration = time.time() - extraction_start
        print(f"Extracted {len(faces)} faces in {extraction_duration:.2f} seconds")
        
        # Validate the number of detected faces
        if len(faces) > 1:
            raise HTTPException(
                status_code=422,
                detail={"type": "MULTIFACE", "message": "Photos with only one face allowed"}
            )
        if len(faces) == 0:
            raise HTTPException(
                status_code=422,
                detail={"type": "NOFACE", "message": "No faces detected in the photo"}
            )
        
        # For the single detected face, get and normalize its embedding
        face = faces[0]
        embedding = face.embedding
        if embedding is None:
            raise HTTPException(
                status_code=422,
                detail={"type": "NOEMBEDDING", "message": "No embedding found in the face"}
            )
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise HTTPException(
                status_code=422,
                detail={"type": "ZERO_NORM", "message": "Encountered zero-norm embedding"}
            )
        # Normalize embedding and explicitly convert each element to a native float
        normalized_embedding = [float(x) for x in (embedding / norm).astype(np.float32).tolist()]
        
        # Return the normalized embedding vector in a JSON response
        return {"embedding": normalized_embedding}
