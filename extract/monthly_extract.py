import os
import boto3
import datetime as dt

bucket = os.environ["S3_BUCKET"]
region = os.environ.get("AWS_REGION", "us-east-1")

s3 = boto3.client("s3", region_name=region)

now = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
key = "raw/test/hello.txt"
body = f"Hello from GitHub Actions at {now}\n".encode("utf-8")

s3.put_object(
    Bucket=bucket,
    Key=key,
    Body=body,
    ContentType="text/plain",
)

print(f"Uploaded s3://{bucket}/{key}")
