# Human Recognition

A face-recognition and face-embedding processing pipeline built around **InsightFace**, **Modal**, **AWS Lambda**, **Amazon S3**, **Amazon SQS**, **PostgreSQL**, and **pgvector**.

The project extracts numerical face embeddings from images and stores them in PostgreSQL so that faces can later be compared using vector similarity search.

> **Status:** Experimental / research project
> **Primary language:** Python
> **Computer vision:** InsightFace / `antelopev2`
> **Vector database:** PostgreSQL + pgvector
> **Cloud compute:** Modal + AWS Lambda

---

## Overview

The core purpose of this project is to turn images containing human faces into machine-readable **face embeddings**.

A face embedding is a numerical vector representing the visual characteristics of a detected face. Once embeddings are stored, a new face can be converted into an embedding and compared against previously stored embeddings using vector similarity.

The project explores two related workflows:

1. **Batch face processing** — images are processed asynchronously and their embeddings are stored in PostgreSQL.
2. **Face embedding API** — an image can be submitted to an HTTP endpoint and receive its normalized face embedding.

The project also contains earlier experiments using different face-recognition approaches, including DeepFace/FaceNet, before moving toward InsightFace and `antelopev2`.

---

## Architecture

The production-oriented pipeline is designed around asynchronous image processing.

```text
                       ┌─────────────────┐
                       │   Application   │
                       │ uploads image   │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Amazon S3       │
                       │ source image    │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Amazon SQS      │
                       │ image event     │
                       └────────┬────────┘
                                │
                                ▼
                 ┌──────────────────────────┐
                 │ ImageModalBridgeLambda   │
                 └────────────┬─────────────┘
                              │
                 ┌────────────┴─────────────┐
                 │                          │
                 ▼                          ▼
       ┌──────────────────┐       ┌──────────────────┐
       │ Watermark Lambda │       │ Modal            │
       │                  │       │ Face Processor   │
       └────────┬─────────┘       └────────┬─────────┘
                │                          │
                ▼                          ▼
             S3 bucket              InsightFace
                                     antelopev2
                                          │
                                          ▼
                                   Face embeddings
                                          │
                                          ▼
                                   PostgreSQL
                                     pgvector
```

The Lambda bridge receives SQS messages containing S3 events, generates temporary presigned URLs for the images, invokes the watermarking Lambda, and asynchronously starts the Modal face-extraction function.

---

## Face Recognition Pipeline

The main recognition pipeline uses:

* **InsightFace**
* **antelopev2**
* **ONNX Runtime**
* **OpenCV**
* **NumPy**
* **PostgreSQL**
* **pgvector**

For every image:

1. Download the image.
2. Decode it using OpenCV.
3. Detect faces.
4. Generate a face embedding for every detected face.
5. Normalize the embedding.
6. Store the embedding in PostgreSQL.

The CPU implementation processes images concurrently during download and uses PostgreSQL's `COPY` mechanism for efficient bulk insertion.

The GPU implementation runs the InsightFace model using CUDA on an H100 GPU.

---

## Face Embeddings

The project uses InsightFace's `antelopev2` model to generate face embeddings.

The resulting embedding is normalized before storage:

```python
norm = np.linalg.norm(embedding)

normalized_embedding = (
    embedding / norm
).astype(np.float32)
```

Normalization makes the embeddings suitable for vector similarity comparisons.

The embeddings are stored as vectors in PostgreSQL.

A representative schema used during experimentation is:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE faces (
    id SERIAL PRIMARY KEY,
    photo_key TEXT NOT NULL,
    embedding VECTOR(512) NOT NULL
);
```

---

## Vector Similarity Search

The project experiments with PostgreSQL's `pgvector` extension and HNSW indexes.

An example index configuration is:

```sql
CREATE INDEX faces_embedding_idx
ON faces
USING hnsw (embedding vector_l2_ops)
WITH (
    m = 16,
    ef_construction = 200
);
```

Similarity searches can then be performed directly inside PostgreSQL.

For example:

```sql
SELECT
    name,
    (embedding <=> %s::vector) AS distance
FROM embeddings
WHERE (embedding <=> %s::vector) < %s
ORDER BY distance ASC
LIMIT %s;
```

The experimental search code also configures:

```sql
SET hnsw.ef_search = 100;
```

This allows PostgreSQL to perform approximate nearest-neighbor searches without requiring a separate vector database.

---

## Components

### `lambdas/`

Contains AWS Lambda functions used to process and orchestrate image processing.

#### `ImageCompressorLambda.py`

Creates a low-quality JPEG version of an image.

The function:

1. Receives an S3/SQS event.
2. Downloads the original image from S3.
3. Converts it to RGB.
4. Compresses it as JPEG with quality `30`.
5. Uploads the resulting image to a destination S3 bucket.

The destination bucket is configured using:

```text
DEST_BUCKET_LOW_QUALITY
```

---

#### `ImageWatermarkGeneratorLambda.py`

Creates a watermarked version of an image.

The watermark is loaded from:

```text
utils/watermark.png
```

in the configured low-resolution S3 bucket.

The watermark is cached between Lambda invocations within the same warm execution environment.

The resulting image is written to the configured watermarked-image bucket.

Required environment variables include:

```text
WATERMARKED_BUCKET_NAME
LOW_RESOLUTION_BUCKET_NAME
```

---

#### `ImageModalBridgeLambda.py`

Acts as the bridge between AWS and Modal.

It:

1. Receives SQS messages.
2. Extracts the S3 bucket and image keys.
3. Invokes the watermark Lambda asynchronously.
4. Generates temporary S3 presigned URLs.
5. Sends the image list to Modal.
6. Starts the Modal face-processing function asynchronously.

Configuration is supplied through environment variables rather than hardcoded credentials.

Expected configuration includes:

```text
WATERMARK_LAMBDA_NAME
MODAL_APP_NAME
MODAL_FUNCTION_NAME
MODAL_TOKEN_ID
MODAL_TOKEN_SECRET
```

---

## `modal/`

Contains the Modal implementations responsible for running InsightFace.

### `face_extraction_cpu.py`

CPU-based face extraction implementation.

The Modal container installs:

```text
insightface==0.7.3
opencv-python-headless
requests
numpy
psycopg2-binary
onnxruntime
```

The InsightFace model is loaded once when the Modal container starts.

The PostgreSQL connection is also established during container initialization and reused for subsequent requests.

Images are downloaded concurrently using a `ThreadPoolExecutor`.

Face extraction itself is performed sequentially.

Embeddings are inserted into PostgreSQL in batches using `COPY`.

---

### `face_extraction_gpu.py`

GPU-accelerated face extraction implementation.

The function is configured to use:

```text
H100
```

and installs:

```text
torch
insightface==0.7.3
opencv-python-headless
requests
numpy
psycopg2-binary
onnxruntime-gpu
```

The implementation expects the `antelopev2` model files to exist in a Modal Volume.

The required model files include:

```text
scrfd_10g_bnkps.onnx
glintr100.onnx
genderage.onnx
```

The model is initialized using CUDA:

```python
FaceAnalysis(
    name="antelopev2",
    providers=["CUDAExecutionProvider"]
)
```

---

### `face_extraction_endpoint_cpu.py`

Provides an HTTP endpoint for extracting a single face embedding.

The endpoint accepts an image using:

```text
multipart/form-data
```

with the field:

```text
file
```

Authentication is performed using the `AuthorizationToken` HTTP header.

The expected token is provided through:

```text
AUTHORIZATION_TOKEN
```

The endpoint requires exactly one face in the image.

Possible errors include:

* `401 Unauthorized`
* `NOFACE`
* `MULTIFACE`
* `NOEMBEDDING`
* `ZERO_NORM`
* `INVALID_IMAGE`

On success, the response contains:

```json
{
  "embedding": [
    0.0123,
    -0.0456,
    ...
  ]
}
```

---

## `experimentation/`

Contains experiments performed while evaluating the face-recognition and vector-search architecture.

### `vector_db_search/`

Contains experiments for:

* extracting embeddings
* storing embeddings in PostgreSQL
* creating HNSW indexes
* performing similarity searches

These scripts represent the experimental stage of the project rather than a polished application interface.

---

## Dataset

The repository contains a `people_dataset/` directory containing a large collection of face images.

The dataset contains images named according to the person represented in each image, for example:

```text
Angelina_Jolie_1.jpg
Angelina_Jolie_2.jpg
Angelina_Jolie_3.jpg
...
```

The repository currently contains more than 1,000 entries in this directory.

### Important

The dataset should be treated separately from the source code.

Before redistributing or using this repository commercially, verify:

* image copyright
* dataset licensing
* permission to redistribute the images
* privacy implications
* biometric-data regulations
* applicable terms of service of the original image sources

Do not assume that publicly available images are automatically licensed for redistribution or biometric processing.

---

## Installation

The project does not currently provide a single packaged installation workflow.

The components have different runtime requirements.

### Python

A modern Python environment is recommended.

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies required by the specific component you want to run.

---

## Local PostgreSQL + pgvector

For vector-search experiments, PostgreSQL must have the `vector` extension installed.

Enable it with:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

A basic table can be created with:

```sql
CREATE TABLE faces (
    id SERIAL PRIMARY KEY,
    photo_key TEXT NOT NULL,
    embedding VECTOR(512) NOT NULL
);
```

An HNSW index can then be created:

```sql
CREATE INDEX faces_embedding_idx
ON faces
USING hnsw (embedding vector_l2_ops)
WITH (
    m = 16,
    ef_construction = 200
);
```

---

## Configuration

Credentials and infrastructure configuration should be supplied through environment variables or the corresponding cloud provider's secret-management system.

Do **not** place credentials directly in source code.

### Modal

The Modal implementation expects a secret named:

```text
face-extraction-secrets
```

The secret should contain the sensitive configuration required by the Modal workload.

The application also uses a Modal Volume named:

```text
insightface-models
```

which contains the InsightFace model files.

---

## AWS

The Lambda components use the AWS SDK (`boto3`).

AWS credentials should be provided using AWS's normal credential provider mechanisms, such as:

* IAM roles
* Lambda execution roles
* environment variables
* AWS credential profiles

The application should not contain hardcoded AWS access keys.

---

## End-to-End Processing

A typical production flow is:

### 1. Upload

An application uploads an original image to S3.

### 2. Event

The S3 event is delivered through SQS.

### 3. Lambda bridge

`ImageModalBridgeLambda` receives the event.

### 4. Image processing

The bridge invokes the watermark Lambda.

### 5. Presigned URL

The bridge generates a temporary URL for the original image.

### 6. Modal

The Modal workload downloads the image using the presigned URL.

### 7. Face detection

InsightFace detects all faces in the image.

### 8. Embedding generation

An embedding is generated for each detected face.

### 9. Normalization

The embedding is normalized before storage.

### 10. PostgreSQL

The embedding is stored in PostgreSQL.

### 11. Search

A future image can be converted into an embedding and compared against the stored vectors using pgvector.

---

## Why PostgreSQL + pgvector?

The project uses PostgreSQL as both the application's relational database and its vector store.

This provides several advantages:

* no separate vector database is required
* embeddings can live alongside normal application data
* SQL can combine vector similarity with relational filters
* PostgreSQL transactions can include embedding writes
* HNSW provides approximate nearest-neighbor search
* operational complexity is lower than maintaining a separate vector infrastructure

For an application that already uses PostgreSQL, this is a particularly attractive architecture.

---

## Performance Considerations

The project contains several optimizations for image-processing workloads.

### Concurrent downloads

The CPU implementation downloads multiple images concurrently using:

```python
ThreadPoolExecutor(max_workers=5)
```

This prevents network latency from completely serializing the pipeline.

### Model initialization

The InsightFace model is initialized during container startup rather than for every request.

This reduces repeated model-loading overhead.

### PostgreSQL COPY

The CPU implementation uses PostgreSQL's `COPY` mechanism to insert multiple embeddings efficiently.

This is significantly more appropriate for bulk embedding ingestion than issuing one SQL transaction per embedding.

### GPU processing

The GPU implementation uses Modal's H100 GPU and CUDA-enabled ONNX Runtime for accelerated inference.

---

## Face Matching

Once a face has been converted into an embedding, it can be compared against stored embeddings.

Conceptually:

```text
Input image
     │
     ▼
Face detection
     │
     ▼
Face embedding
     │
     ▼
Vector similarity search
     │
     ▼
Nearest faces
     │
     ▼
Similarity / distance
```

The exact similarity threshold should be calibrated against the specific model, dataset, image quality, and desired false-positive / false-negative tradeoff.

A threshold should therefore not be treated as a universal constant.

---

## Security

This project processes biometric information and should therefore be treated as security-sensitive infrastructure.

Important considerations include:

* Never commit database passwords.
* Never commit AWS access keys.
* Never commit Modal tokens.
* Never commit API authentication tokens.
* Use short-lived presigned URLs where possible.
* Restrict S3 bucket permissions.
* Restrict database credentials to the minimum required privileges.
* Protect face embeddings as sensitive data.
* Encrypt data at rest and in transit.
* Avoid logging full presigned URLs.
* Avoid logging sensitive image information.
* Implement authentication and authorization around production endpoints.
* Establish retention and deletion policies for biometric data.

The repository intentionally reads sensitive configuration from environment variables and Modal secrets rather than embedding those values in the application.

---

## Repository Structure

```text
human_recognition/
│
├── experimentation/
│   └── vector_db_search/
│       ├── face_embedding_vector_db_extraction.py
│       └── face_embedding_vector_db_search.py
│
├── lambdas/
│   ├── ImageCompressorLambda.py
│   ├── ImageModalBridgeLambda.py
│   └── ImageWatermarkGeneratorLambda.py
│
├── modal/
│   ├── face_extraction_cpu.py
│   ├── face_extraction_endpoint_cpu.py
│   ├── face_extraction_gpu.py
│   ├── insight_face_init_test.py
│   └── modal_test.py
│
├── people_dataset/
│   └── ...
│
├── .gitignore
└── README.md
```

---

## Technology Stack

| Component                    | Technology              |
| ---------------------------- | ----------------------- |
| Language                     | Python                  |
| Face detection / recognition | InsightFace             |
| Face model                   | antelopev2              |
| Model runtime                | ONNX Runtime            |
| GPU runtime                  | ONNX Runtime GPU / CUDA |
| Image processing             | OpenCV / Pillow         |
| Numerical processing         | NumPy                   |
| Database                     | PostgreSQL              |
| Vector search                | pgvector                |
| ANN index                    | HNSW                    |
| Cloud storage                | Amazon S3               |
| Messaging                    | Amazon SQS              |
| Serverless processing        | AWS Lambda              |
| GPU/serverless compute       | Modal                   |
| HTTP API                     | FastAPI                 |

---

## Project History

The repository began as a collection of face-recognition experiments.

Earlier experiments explored DeepFace/FaceNet and direct PostgreSQL vector storage.

The project subsequently moved toward InsightFace and `antelopev2`, followed by:

* PostgreSQL vector search experiments
* HNSW indexing
* Modal-based model execution
* CPU face extraction
* GPU face extraction
* an HTTP embedding endpoint
* asynchronous AWS Lambda orchestration
* image compression
* image watermarking

The repository's commit history reflects this evolution from experimentation toward a cloud-based processing pipeline.

---

## Current State

This repository should be considered a **prototype / infrastructure experiment rather than a complete standalone application**.

It contains the core building blocks required for a face-recognition processing pipeline, but it does not currently provide:

* a unified CLI
* a complete dependency lockfile
* infrastructure-as-code
* automated deployment
* comprehensive automated tests
* production API documentation
* database migrations
* formal configuration management
* a complete local development environment

The individual components are more useful as implementation references or as parts of a larger application.

---

## Future Improvements

Potential improvements include:

* Add a proper dependency management system.
* Add environment-specific configuration.
* Add infrastructure-as-code for AWS and Modal.
* Add database migrations.
* Add automated tests for face extraction and matching.
* Add structured logging.
* Add monitoring and tracing.
* Add retry/dead-letter handling to asynchronous processing.
* Add idempotency to embedding ingestion.
* Add explicit image and embedding retention policies.
* Add authentication and rate limiting to the embedding API.
* Add benchmark scripts for CPU vs GPU inference.
* Benchmark HNSW parameters against real-world datasets.
* Add evaluation metrics such as precision, recall, FAR, and FRR.
* Separate experimental code from production code.
* Remove datasets and personal/local artifacts from the source repository.
* Document model licensing and dataset provenance.

---

## Disclaimer

This project is intended for experimentation and research into face detection, face embeddings, and vector similarity search.

Face recognition is a sensitive biometric technology. Before deploying this system in a real-world application, evaluate applicable privacy, biometric-data, security, and data-protection requirements.

---

## License

No explicit project license is currently provided.

If this project is intended to be distributed or used publicly, add an appropriate `LICENSE` file and document the licenses of the underlying models and datasets.
