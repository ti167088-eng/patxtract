"""
Secure API Management

Handles secure loading and management of API keys from environment variables.
"""

import os
import logging
from dotenv import load_dotenv
from typing import Optional

logger = logging.getLogger(__name__)

class SecureConfig:
    """
    Secure configuration manager that loads API keys from environment variables.
    """

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize the secure configuration.

        Args:
            env_file: Optional path to .env file. If None, will look in default locations.
        """
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()  # Load from default .env file
        logger.info("Environment variables loaded securely")

    def get_api_key(self, provider: str) -> str:
        """
        Get API key for a specific provider from environment variables.

        Args:
            provider: The name of the provider (e.g., 'openrouter', 'openai')

        Returns:
            API key as string

        Raises:
            ValueError: If API key is not found
        """
        # Try multiple possible environment variable names
        possible_keys = [
            f"{provider.upper()}_API_KEY",
            f"{provider.upper()}_KEY",
            provider.upper(),
            "API_KEY"  # Fallback to generic API_KEY
        ]

        for key_name in possible_keys:
            api_key = os.getenv(key_name)
            if api_key:
                logger.debug(f"Found API key for {provider} using environment variable {key_name}")
                return api_key.strip()

        raise ValueError(f"API key for {provider} not found in environment variables. "
                        f"Tried: {', '.join(possible_keys)}")

    def get_openrouter_key(self) -> str:
        """
        Get OpenRouter API key.

        Returns:
            OpenRouter API key as string
        """
        return self.get_api_key("openrouter")

    def get_base_url(self, provider: str = "openrouter") -> str:
        """
        Get base URL for API endpoint.

        Args:
            provider: The provider name

        Returns:
            Base URL as string
        """
        if provider.lower() == "openrouter":
            return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def get_model_config(self, model_name: str) -> dict:
        """
        Get model-specific configuration.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model configuration
        """
        default_config = {
            "temperature": 0.1,
            "max_tokens": 4000,
            "timeout": 30
        }

        # Override with model-specific settings from environment if available
        model_config = default_config.copy()

        temp_key = f"{model_name.upper().replace('-', '_')}_TEMPERATURE"
        tokens_key = f"{model_name.upper().replace('-', '_')}_MAX_TOKENS"
        timeout_key = f"{model_name.upper().replace('-', '_')}_TIMEOUT"

        if os.getenv(temp_key):
            try:
                model_config["temperature"] = float(os.getenv(temp_key))
            except ValueError:
                logger.warning(f"Invalid temperature value for {model_name}: {os.getenv(temp_key)}")

        if os.getenv(tokens_key):
            try:
                model_config["max_tokens"] = int(os.getenv(tokens_key))
            except ValueError:
                logger.warning(f"Invalid max_tokens value for {model_name}: {os.getenv(tokens_key)}")

        if os.getenv(timeout_key):
            try:
                model_config["timeout"] = int(os.getenv(timeout_key))
            except ValueError:
                logger.warning(f"Invalid timeout value for {model_name}: {os.getenv(timeout_key)}")

        return model_config

    def validate_config(self) -> bool:
        """
        Validate that required configuration is present.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Try to get OpenRouter key
            self.get_openrouter_key()
            logger.info("Configuration validation successful")
            return True
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            return False