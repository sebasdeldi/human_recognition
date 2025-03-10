import io
import os
import boto3
import json
import time
import urllib.parse
from PIL import Image

from PIL import features
print(Image.__version__)
print(features.check("libjpeg_turbo"))

s3 = boto3.client("s3")

# Load destination bucket from environment variable
DEST_BUCKET_LOW_QUALITY = os.getenv("DEST_BUCKET_LOW_QUALITY")
if not DEST_BUCKET_LOW_QUALITY:
    raise RuntimeError("❌ DEST_BUCKET_LOW_QUALITY environment variable is not set.")

def upload_to_s3(bucket, key, data, content_type):
    """Uploads the image to S3."""
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)

def lambda_handler(event, context):
    records = event.get("Records", [])
    print(f"Received {len(records)} messages.")

    for record in records:
        message = json.loads(record["body"])
        s3_bucket = message["Records"][0]["s3"]["bucket"]["name"]
        raw_key = message["Records"][0]["s3"]["object"]["key"]
        s3_key = urllib.parse.unquote(raw_key)

        try:
            start_time = time.time()

            # Download image directly from S3 without an intermediate buffer
            s3_response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
            image_data = s3_response["Body"].read()
            content_type = s3_response.get("ContentType", "image/jpeg")

            # Open the image
            image = Image.open(io.BytesIO(image_data))
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Generate Low-Quality Image
            low_res_start = time.time()
            low_res_buffer = io.BytesIO()
            image.save(low_res_buffer, format="JPEG", quality=30, optimize=True)
            low_res_data = low_res_buffer.getvalue()
            low_res_buffer.close()
            low_res_time = time.time() - low_res_start
            print(f"⏳ Low-res generation took {low_res_time:.2f}s")

            # Upload image
            upload_start = time.time()
            upload_to_s3(DEST_BUCKET_LOW_QUALITY, s3_key, low_res_data, content_type)
            upload_time = time.time() - upload_start

            total_time = time.time() - start_time
            print(f"✅ Processed {s3_key} | Total: {total_time:.2f}s | Low-Res Gen: {low_res_time:.2f}s | Upload: {upload_time:.2f}s")

            # Cleanup
            image.close()
            del image, image_data, low_res_data

        except Exception as e:
            print(f"❌ Error processing {s3_key}: {e}")
            return {"statusCode": 500, "body": f"Error processing {s3_key}: {str(e)}"}

    return {"statusCode": 200, "body": "Processed images successfully"}
