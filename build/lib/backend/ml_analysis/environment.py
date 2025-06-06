# backend/ml_analysis/environment.py

import os
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Gemini API configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def check_api_configuration():
    """Check if necessary API keys are configured."""
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is not set. Some features may not work.")
        return False
    return True

def get_api_key(key_name):
    """Safely retrieve API key."""
    key = os.environ.get(key_name)
    if not key:
        logger.warning(f"{key_name} is not set.")
    return key