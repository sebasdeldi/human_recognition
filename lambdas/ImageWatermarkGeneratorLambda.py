import io
import os
import boto3
import json
import time
from PIL import Image, ImageEnhance

# Initialize S3 client
s3 = boto3.client("s3")

# Destination bucket for watermarked images
DEST_BUCKET_WATERMARKED = os.environ["WATERMARKED_BUCKET_NAME"]

# Watermark Info
WATERMARK_S3_BUCKET = os.environ["LOW_RESOLUTION_BUCKET_NAME"]
WATERMARK_S3_KEY = "utils/watermark.png"

# Global watermark (cached per cold start)
WATERMARK_IMAGE = None

def load_watermark():
    """
    Loads the watermark image from S3 into memory.
    This function ensures the watermark is only loaded once per cold start.
    
    Returns:
        Image: A PIL Image object of the watermark with reduced opacity.
    """
    global WATERMARK_IMAGE
    if WATERMARK_IMAGE is None:
        print("🔹 Loading watermark from S3 (cold start)...")
        watermark_obj = s3.get_object(Bucket=WATERMARK_S3_BUCKET, Key=WATERMARK_S3_KEY)
        watermark_image = Image.open(io.BytesIO(watermark_obj["Body"].read())).convert("RGBA")

        # Optimize watermark by reducing opacity
        enhancer = ImageEnhance.Brightness(watermark_image)
        WATERMARK_IMAGE = enhancer.enhance(0.7)  # Reduce opacity to 70%

    return WATERMARK_IMAGE

def apply_watermark(image):
    """
    Applies a centered watermark to the given image.
    
    Args:
        image (PIL.Image): The input image to be watermarked.
    
    Returns:
        PIL.Image: The watermarked image.
    """
    image = image.convert("RGBA")
    watermark = load_watermark()

    # Center the watermark
    x = (image.width - watermark.width) // 2
    y = (image.height - watermark.height) // 2

    # Blend watermark with image
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    overlay.paste(watermark, (x, y), watermark)
    return Image.alpha_composite(image, overlay).convert("RGB")

def upload_to_s3(bucket, key, image):
    """
    Uploads an image to an S3 bucket.
    
    Args:
        bucket (str): The name of the S3 bucket.
        key (str): The S3 object key (file path in S3).
        image (PIL.Image): The image to be uploaded.
    """
    with io.BytesIO() as output_buffer:
        image.save(output_buffer, format="JPEG", quality=30)
        output_data = output_buffer.getvalue()
        s3.put_object(Bucket=bucket, Key=key, Body=output_data, ContentType="image/jpeg")

def lambda_handler(event, context):
    """
    AWS Lambda handler function to process images, apply watermark, and upload them back to S3.
    
    Args:
        event (dict): The event data passed to the function, containing:
            - bucket (str): The source S3 bucket name.
            - keys (list): List of S3 object keys to process.
        context (LambdaContext): AWS Lambda runtime information.
    
    Returns:
        dict: A response indicating success or failure.
    """
    bucket = event.get("bucket")
    keys = event.get("keys", [])

    if not bucket or not keys:
        print("❌ Invalid event: Missing 'bucket' or 'keys'.")
        return {"statusCode": 400, "message": "Invalid input format"}

    print(f"🔹 Processing {len(keys)} images from {bucket}")

    for s3_key in keys:
        try:
            start_time = time.time()

            # Fetch image from S3 (batch optimized)
            image_obj = s3.get_object(Bucket=bucket, Key=s3_key)
            image = Image.open(io.BytesIO(image_obj["Body"].read())).convert("RGB")

            # Apply watermark
            watermark_start = time.time()
            watermarked_image = apply_watermark(image)
            print(f"⏳ Watermark applied in {time.time() - watermark_start:.2f}s")

            # Upload to S3
            upload_start = time.time()
            upload_to_s3(DEST_BUCKET_WATERMARKED, s3_key, watermarked_image)
            print(f"⏳ Upload took {time.time() - upload_start:.2f}s")

            # Cleanup memory
            del image, watermarked_image

            print(f"✅ Processed {s3_key} in {time.time() - start_time:.2f}s")

        except Exception as e:
            print(f"❌ Error processing {s3_key}: {e}")
            return {"statusCode": 500, "message": f"Error processing {s3_key}: {str(e)}"}

    return {"statusCode": 200, "message": "Processed images successfully"}
