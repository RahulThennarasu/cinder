# Modified version of backend/auth/auth.py

import os
import hashlib
import hmac
import time
import logging
from typing import Optional, Dict, Any, List
import json

# Path to the file storing valid keys
KEYS_FILE = os.path.join(os.path.dirname(__file__), 'valid_keys.json')

# In-memory cache of valid API keys
_API_KEY_CACHE: Dict[str, Dict[str, Any]] = {}

# Master key for admin/development use
_MASTER_KEY = os.environ.get("CINDER_MASTER_KEY", "cinder_master_key_change_me")

def _load_valid_keys() -> Dict[str, Dict[str, Any]]:
    """Load valid keys from the keys file."""
    if os.path.exists(KEYS_FILE):
        try:
            with open(KEYS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading keys file: {e}")
    return {}

def _save_valid_keys(keys: Dict[str, Dict[str, Any]]) -> bool:
    """Save valid keys to the keys file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(KEYS_FILE), exist_ok=True)
        
        with open(KEYS_FILE, 'w') as f:
            json.dump(keys, f, indent=2)
        return True
    except Exception as e:
        logging.error(f"Error saving keys file: {e}")
        return False

def validate_api_key(api_key: Optional[str]) -> bool:
    """
    Validate if an API key is valid.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        bool: True if the API key is valid, False otherwise
    """
    if not api_key:
        return False
    
    # Check cache first
    if api_key in _API_KEY_CACHE:
        # Check if the cached key is expired
        if _API_KEY_CACHE[api_key].get("expires_at", 0) > time.time():
            return True
        else:
            # Remove expired key
            del _API_KEY_CACHE[api_key]
    
    # For development: allow master key for testing
    if api_key == _MASTER_KEY:
        _API_KEY_CACHE[api_key] = {
            "valid": True,
            "expires_at": time.time() + (365 * 24 * 60 * 60),  # 1 year
            "permissions": ["all"]
        }
        return True
    
    # Check valid keys file
    valid_keys = _load_valid_keys()
    if api_key in valid_keys:
        # Check if key is expired
        if valid_keys[api_key].get("expires_at", 0) > time.time():
            # Add to cache
            _API_KEY_CACHE[api_key] = valid_keys[api_key]
            return True
        else:
            # Remove expired key
            del valid_keys[api_key]
            _save_valid_keys(valid_keys)
    
    # As a fallback, check environment variable
    env_valid_keys = os.environ.get("CINDER_VALID_KEYS", "").split(",")
    if api_key in env_valid_keys:
        _API_KEY_CACHE[api_key] = {
            "valid": True,
            "expires_at": time.time() + (30 * 24 * 60 * 60),  # 30 days
            "permissions": ["basic"]
        }
        return True
    
    return False

def generate_api_key(user_id: str) -> str:
    """
    Generate a new API key for a user and add it to the valid keys list.
    
    Args:
        user_id: Unique identifier for the user
        
    Returns:
        str: Generated API key
    """
    timestamp = str(int(time.time()))
    
    # Create a unique key based on user_id, timestamp and a secret
    secret = os.environ.get("CINDER_API_SECRET", "change_this_secret_key")
    message = f"{user_id}:{timestamp}"
    
    # Generate HMAC using SHA-256
    signature = hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Create the formatted API key
    api_key = f"cinder_{timestamp}_{signature[:32]}"
    
    # Add the key to the list of valid keys
    valid_keys = _load_valid_keys()
    valid_keys[api_key] = {
        "user_id": user_id,
        "created_at": time.time(),
        "expires_at": time.time() + (365 * 24 * 60 * 60),  # 1 year
        "permissions": ["basic"]
    }
    _save_valid_keys(valid_keys)
    
    return api_key

def get_key_permissions(api_key: Optional[str]) -> List[str]:
    """
    Get the permissions associated with an API key.
    
    Args:
        api_key: The API key
        
    Returns:
        list: List of permission strings
    """
    if not api_key:
        return []
    
    if api_key in _API_KEY_CACHE:
        return _API_KEY_CACHE[api_key].get("permissions", [])
    
    if validate_api_key(api_key):
        return _API_KEY_CACHE[api_key].get("permissions", [])
    
    return []

def revoke_api_key(api_key: str) -> bool:
    """
    Revoke an API key.
    
    Args:
        api_key: The API key to revoke
        
    Returns:
        bool: True if the key was revoked, False otherwise
    """
    # Remove from cache
    if api_key in _API_KEY_CACHE:
        del _API_KEY_CACHE[api_key]
    
    # Remove from valid keys file
    valid_keys = _load_valid_keys()
    if api_key in valid_keys:
        del valid_keys[api_key]
        return _save_valid_keys(valid_keys)
    
    return False

def list_valid_keys() -> List[Dict[str, Any]]:
    """
    List all valid API keys.
    
    Returns:
        list: List of key information dictionaries
    """
    valid_keys = _load_valid_keys()
    
    # Format keys for display
    result = []
    for key, info in valid_keys.items():
        # Don't include the full key for security
        masked_key = f"{key[:10]}...{key[-5:]}"
        result.append({
            "key": masked_key,
            "user_id": info.get("user_id", "unknown"),
            "created_at": info.get("created_at", 0),
            "expires_at": info.get("expires_at", 0),
            "permissions": info.get("permissions", [])
        })
    
    return result