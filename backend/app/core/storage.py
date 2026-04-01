import asyncio
import io
from functools import partial

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.core.config import settings
from app.core.exceptions import StorageError
from app.core.logging import get_logger

logger = get_logger(__name__)


def _get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        config=Config(signature_version="s3v4"),
    )


async def upload_file(file_bytes: bytes, s3_key: str, content_type: str) -> str:
    """Upload bytes to S3/MinIO. Returns s3_key on success, raises StorageError on failure."""
    loop = asyncio.get_running_loop()

    def _upload() -> str:
        client = _get_s3_client()
        client.put_object(
            Bucket=settings.s3_bucket,
            Key=s3_key,
            Body=io.BytesIO(file_bytes),
            ContentType=content_type,
        )
        return s3_key

    try:
        result = await loop.run_in_executor(None, _upload)
        logger.info("File uploaded to S3", s3_key=s3_key, content_type=content_type)
        return result
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "unknown")
        logger.error("S3 upload failed", s3_key=s3_key, error_code=error_code, exc_info=exc)
        raise StorageError(f"Failed to upload file '{s3_key}': {error_code}") from exc
    except Exception as exc:
        logger.error("S3 upload unexpected error", s3_key=s3_key, exc_info=exc)
        raise StorageError(f"Unexpected error uploading '{s3_key}'") from exc


async def get_presigned_url(s3_key: str, expires_in: int = 3600) -> str:
    """Generate a presigned GET URL for an S3 object. Raises StorageError on failure."""
    loop = asyncio.get_running_loop()

    def _presign() -> str:
        client = _get_s3_client()
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": s3_key},
            ExpiresIn=expires_in,
        )

    try:
        url = await loop.run_in_executor(None, _presign)
        logger.debug("Presigned URL generated", s3_key=s3_key, expires_in=expires_in)
        return url
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "unknown")
        logger.error("S3 presign failed", s3_key=s3_key, error_code=error_code, exc_info=exc)
        raise StorageError(f"Failed to generate presigned URL for '{s3_key}': {error_code}") from exc
    except Exception as exc:
        logger.error("S3 presign unexpected error", s3_key=s3_key, exc_info=exc)
        raise StorageError(f"Unexpected error generating presigned URL for '{s3_key}'") from exc


async def delete_file(s3_key: str) -> None:
    """Delete an object from S3/MinIO. Raises StorageError on failure."""
    loop = asyncio.get_running_loop()

    def _delete() -> None:
        client = _get_s3_client()
        client.delete_object(Bucket=settings.s3_bucket, Key=s3_key)

    try:
        await loop.run_in_executor(None, _delete)
        logger.info("File deleted from S3", s3_key=s3_key)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "unknown")
        logger.error("S3 delete failed", s3_key=s3_key, error_code=error_code, exc_info=exc)
        raise StorageError(f"Failed to delete file '{s3_key}': {error_code}") from exc
    except Exception as exc:
        logger.error("S3 delete unexpected error", s3_key=s3_key, exc_info=exc)
        raise StorageError(f"Unexpected error deleting '{s3_key}'") from exc


async def ensure_bucket_exists() -> None:
    """Create the configured S3 bucket if it does not already exist. Called at startup."""
    loop = asyncio.get_running_loop()

    def _ensure() -> None:
        client = _get_s3_client()
        try:
            client.head_bucket(Bucket=settings.s3_bucket)
            logger.info("S3 bucket already exists", bucket=settings.s3_bucket)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            # 404 / NoSuchBucket means we need to create it
            if error_code in ("404", "NoSuchBucket"):
                client.create_bucket(Bucket=settings.s3_bucket)
                logger.info("S3 bucket created", bucket=settings.s3_bucket)
            else:
                raise

    try:
        await loop.run_in_executor(None, _ensure)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "unknown")
        logger.error(
            "Failed to ensure S3 bucket exists",
            bucket=settings.s3_bucket,
            error_code=error_code,
            exc_info=exc,
        )
        raise StorageError(
            f"Failed to ensure bucket '{settings.s3_bucket}' exists: {error_code}"
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error ensuring S3 bucket", bucket=settings.s3_bucket, exc_info=exc)
        raise StorageError(
            f"Unexpected error ensuring bucket '{settings.s3_bucket}'"
        ) from exc
