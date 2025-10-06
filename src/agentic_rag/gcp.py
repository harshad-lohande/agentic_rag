# src/agentic_rag/gcp.py

import json
import logging
from functools import lru_cache

from google.cloud import secretmanager
from google.api_core import exceptions as gcp_exceptions

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_secrets(project_id: str, secret_id: str, version_id: str = "latest") -> dict:
    """
    Retrieves a secret from Google Cloud Secret Manager.

    The secret is expected to be a JSON string containing key-value pairs.
    This function is cached to avoid repeated API calls.

    Args:
        project_id: The Google Cloud project ID.
        secret_id: The ID of the secret.
        version_id: The version of the secret to retrieve.

    Returns:
        A dictionary containing the secrets.

    Raises:
        ValueError: If the secret cannot be retrieved or parsed.
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"

    logger.info(f"Attempting to retrieve secret '{name}' from Google Secret Manager.")

    try:
        response = client.access_secret_version(request={"name": name})
        payload = response.payload.data.decode("UTF-8")
        secrets_dict = json.loads(payload)
        logger.info(f"Successfully retrieved and parsed secret '{secret_id}'.")
        return secrets_dict
    except gcp_exceptions.NotFound:
        logger.error(f"Secret '{name}' not found.")
        raise ValueError(f"Secret '{name}' not found.")
    except gcp_exceptions.PermissionDenied:
        logger.error(f"Permission denied for secret '{name}'. Check IAM permissions.")
        raise ValueError(f"Permission denied for secret '{name}'.")
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON from secret '{secret_id}'.")
        raise ValueError(f"The secret value for '{secret_id}' is not valid JSON.")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while fetching secret '{secret_id}': {e}"
        )
        raise ValueError(f"An unexpected error occurred: {e}")
