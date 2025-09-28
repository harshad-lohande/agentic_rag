# src/agentic_rag/aws.py

import json
import logging
from functools import lru_cache

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_secrets(secret_name: str, region_name: str) -> dict:
    """
    Retrieves a secret from AWS Secrets Manager.

    The secret is expected to be a JSON string containing key-value pairs.
    This function is cached to avoid repeated API calls for the same secret
    within the application's lifecycle.

    Args:
        secret_name: The name of the secret in AWS Secrets Manager.
        region_name: The AWS region where the secret is stored.

    Returns:
        A dictionary containing the secrets.

    Raises:
        ValueError: If the secret cannot be retrieved or parsed.
    """
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    logger.info(f"Attempting to retrieve secret '{secret_name}' from AWS Secrets Manager.")

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except NoCredentialsError:
        logger.error("AWS credentials not found. Please configure your credentials.")
        raise ValueError("AWS credentials not configured.")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        logger.error(f"Failed to retrieve secret '{secret_name}'. Error code: {error_code}")
        raise ValueError(f"Cannot retrieve secret '{secret_name}': {error_code}")

    # Secrets Manager can return the secret as a string or binary. We expect a string.
    secret = get_secret_value_response.get('SecretString')
    if not secret:
        logger.error(f"Secret string for '{secret_name}' is empty or not found in the response.")
        raise ValueError(f"Secret string for '{secret_name}' is empty.")

    try:
        secrets_dict = json.loads(secret)
        logger.info(f"Successfully retrieved and parsed secret '{secret_name}'.")
        return secrets_dict
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON from secret '{secret_name}'.")
        raise ValueError(f"The secret string for '{secret_name}' is not valid JSON.")