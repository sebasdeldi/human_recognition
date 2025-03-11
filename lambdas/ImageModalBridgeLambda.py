import json
import boto3
import os
import modal

# -----------------------------------------------------------------------------
# AWS Environment Setup
# -----------------------------------------------------------------------------
lambda_client = boto3.client("lambda")

# Load required environment variables.
WATERMARK_LAMBDA_NAME = os.getenv("WATERMARK_LAMBDA_NAME")
if not WATERMARK_LAMBDA_NAME:
    raise RuntimeError("WATERMARK_LAMBDA_NAME environment variable is not set.")

MODAL_APP_NAME = os.getenv("MODAL_APP_NAME")
if not MODAL_APP_NAME:
    raise RuntimeError("MODAL_APP_NAME environment variable is not set.")

MODAL_FUNCTION_NAME = os.getenv("MODAL_FUNCTION_NAME")
if not MODAL_FUNCTION_NAME:
    raise RuntimeError("MODAL_FUNCTION_NAME environment variable is not set.")

MODAL_TOKEN_ID = os.getenv("MODAL_TOKEN_ID")
if not MODAL_TOKEN_ID:
    raise RuntimeError("MODAL_TOKEN_ID environment variable is not set.")

MODAL_TOKEN_SECRET = os.getenv("MODAL_TOKEN_SECRET")
if not MODAL_TOKEN_SECRET:
    raise RuntimeError("MODAL_TOKEN_SECRET environment variable is not set.")


# -----------------------------------------------------------------------------
# Helper Function to Generate Presigned URLs
# -----------------------------------------------------------------------------
def get_presigned_url(bucket, key, expiration=900):
    """
    Generate a presigned URL for an S3 object valid for 15 minutes (900 seconds).
    
    :param bucket: The S3 bucket name.
    :param key: The object key.
    :param expiration: Time in seconds for the presigned URL to remain valid.
    :return: Presigned URL as a string.
    """
    s3_client = boto3.client("s3")
    url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': key},
        ExpiresIn=expiration
    )
    return url

# -----------------------------------------------------------------------------
# Lambda Handler
# -----------------------------------------------------------------------------
def lambda_handler(event, context):
    """
    Lambda handler that processes SQS events containing S3 records.
    
    It performs two main tasks:
      1. Asynchronously invokes the Watermarked Image Generator Lambda.
      2. Constructs a payload with presigned URLs (valid for 15 minutes) and
         invokes the Modal function asynchronously via the Modal SDK.
    
    :param event: The event payload containing SQS messages with S3 records.
    :param context: The Lambda context object.
    :return: A dict with a status code and message.
    """
    # --- Extract SQS Records ---
    records = event.get("Records", [])
    if not records:
        print("❌ No records received.")
        return {"statusCode": 200, "message": "No records to process"}
    print(f"✅ Received {len(records)} SQS messages.")

    # Parse and extract S3 records from the SQS messages.
    s3_records = []
    for record in records:
        try:
            body = json.loads(record["body"])
            s3_records.extend(body.get("Records", []))
        except Exception as e:
            print(f"⚠️ Failed to parse SQS body: {e}")
    if not s3_records:
        print("❌ No valid S3 records found in SQS messages.")
        return {"statusCode": 400, "message": "Invalid event format"}
    print(f"✅ Extracted {len(s3_records)} S3 event records.")

    # --- Extract Bucket Name and Object Keys ---
    first_record = s3_records[0]
    bucket = first_record["s3"]["bucket"]["name"]
    keys = [r["s3"]["object"]["key"] for r in s3_records if "s3" in r and "object" in r["s3"]]
    if not keys:
        print("❌ No valid S3 keys found.")
        return {"statusCode": 400, "message": "No valid keys to process"}

    # --- Invoke Watermarked Image Generator Lambda ---
    watermark_payload = {"bucket": bucket, "keys": keys}
    print(f"📦 Payload for watermark lambda prepared: {json.dumps(watermark_payload)}")
    try:
        lambda_client.invoke(
            FunctionName=WATERMARK_LAMBDA_NAME,
            InvocationType="Event",  # Asynchronous invocation
            Payload=json.dumps(watermark_payload),
        )
        print(f"✅ Forwarded {len(keys)} image keys to watermark lambda.")
    except Exception as e:
        print(f"❌ Failed to invoke watermark lambda: {e}")
        return {"statusCode": 500, "message": f"Failed to invoke watermark lambda: {str(e)}"}

    # --- Prepare Payload for Modal Function ---
    image_list = []
    for key in keys:
        presigned_url = get_presigned_url(bucket, key, expiration=900)
        image_list.append({
            "photo_key": key,
            "image_url": presigned_url
        })
    print(f"📦 Payload for Modal function prepared with {len(image_list)} images.")

    # --- Asynchronously Invoke the Modal Function via Modal SDK ---
    try:
        modal_function = modal.Function.from_name(MODAL_APP_NAME, MODAL_FUNCTION_NAME)
        function_call = modal_function.spawn({"image_list": image_list})
        print("✅ Successfully invoked Modal function via Modal SDK.")
    except Exception as e:
        print(f"❌ Failed to invoke Modal function: {e}")
        return {"statusCode": 500, "message": f"Failed to invoke Modal function: {str(e)}"}

    return {
        "statusCode": 200,
        "message": f"Processed {len(keys)} image keys and invoked both watermark and modal functions."
    }
