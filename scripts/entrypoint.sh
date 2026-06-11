#!/bin/sh
set -e

# Sync data from S3 if AWS credentials and bucket are configured.
# This runs on every container startup so documents/vector_store stay current.
if [ -n "$S3_BUCKET_NAME" ] && [ -n "$AWS_ACCESS_KEY_ID" ]; then
    echo "Syncing data from s3://$S3_BUCKET_NAME/$S3_DATA_PREFIX ..."
    python - <<'EOF'
import boto3, os
from pathlib import Path

s3 = boto3.client(
    "s3",
    region_name=os.environ.get("AWS_REGION", "us-east-1"),
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)
bucket = os.environ["S3_BUCKET_NAME"]
prefix = os.environ.get("S3_DATA_PREFIX", "data/").rstrip("/") + "/"

paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        # Map s3 key → local path (strip the prefix, keep the rest under data/)
        rel = key[len(prefix):]
        if not rel:
            continue
        local_path = Path("data") / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  {key} -> {local_path}")
        s3.download_file(bucket, key, str(local_path))

print("S3 sync complete.")
EOF
fi

exec "$@"
