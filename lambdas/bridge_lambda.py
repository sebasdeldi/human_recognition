import json
import boto3
import os

# Initialize AWS clients outside the handler for efficiency
lambda_client = boto3.client("lambda")

# Load environment variables once at startup
WATERMARK_LAMBDA_NAME = os.getenv("WATERMARK_LAMBDA_NAME")
if not WATERMARK_LAMBDA_NAME:
    raise RuntimeError("❌ WATERMARK_LAMBDA_NAME environment variable is not set.")

def lambda_handler(event, context):
    records = event.get("Records", [])
    if not records:
        print("❌ No records received.")
        return {"statusCode": 200, "message": "No records to process"}

    print(f"✅ Received {len(records)} SQS messages.")

    # Extract S3 records from SQS messages
    s3_records = []
    for record in records:
        try:
            body = json.loads(record["body"])  # Parse SQS body
            s3_records.extend(body.get("Records", []))  # Extract S3 Records
        except Exception as e:
            print(f"⚠️ Failed to parse SQS body: {e}")

    if not s3_records:
        print("❌ No valid S3 records found in SQS messages.")
        return {"statusCode": 400, "message": "Invalid event format"}

    print(f"✅ Extracted {len(s3_records)} S3 event records.")

    # Extract bucket name from the first valid S3 record
    first_record = s3_records[0]
    bucket = first_record["s3"]["bucket"]["name"]
    
    # Extract object keys efficiently
    keys = [r["s3"]["object"]["key"] for r in s3_records if "s3" in r and "object" in r["s3"]]

    if not keys:
        print("❌ No valid S3 keys found.")
        return {"statusCode": 400, "message": "No valid keys to process"}

    # Prepare the payload for the Watermarked Image Generator Lambda
    payload = {"bucket": bucket, "keys": keys}
    print(f"📦 Payload prepared: {json.dumps(payload)}")

    try:
        # Invoke the Watermarked Image Generator Lambda asynchronously
        response = lambda_client.invoke(
            FunctionName=WATERMARK_LAMBDA_NAME,
            InvocationType="Event",  # Asynchronous invocation
            Payload=json.dumps(payload),
        )
        print(f"✅ Forwarded {len(keys)} image keys to watermark lambda.")
    except Exception as e:
        print(f"❌ Failed to invoke watermark lambda: {e}")
        return {"statusCode": 500, "message": f"Failed to invoke watermark lambda: {str(e)}"}

    return {"statusCode": 200, "message": f"Forwarded {len(keys)} image keys."}
