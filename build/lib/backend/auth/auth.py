# Modified version of backend/auth/auth.py

import os
import hashlib
import hmac
import time
import logging
from typing import Optional, Dict, Any, List
import json

import firebase_admin
from firebase_admin import credentials, firestore

# Path to the file storing valid keys
KEYS_FILE = os.path.join(os.path.dirname(__file__), 'valid_keys.json')

# In-memory cache of valid API keys
_API_KEY_CACHE: Dict[str, Dict[str, Any]] = {}

# Master key for admin/development use
_MASTER_KEY = os.environ.get("CINDER_MASTER_KEY", "cinder_master_key_change_me")

try:
    cred_path = os.path.join(os.path.dirname(__file__), 'firebase-credentials.json')
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    print(f"Firebase initialization error: {e}")
    db = None

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

def validate_api_key(api_key):
    """Validate an API key against Firebase database."""
    if not api_key:
        return False
    
    if not db:
        print("Firebase not initialized, skipping API key validation")
        return True  # For development, assume valid if Firebase not initialized
    
    try:
        # Query Firestore for the API key
        query = db.collection("api_keys").where("key", "==", api_key).limit(1)
        results = list(query.stream())
        
        if not results:
            return False
        
        key_doc = results[0]
        key_data = key_doc.to_dict()
        
        # Check if the key is active
        if not key_data.get("active", True):  # Default to active if not specified
            return False
        
        # Check if expired
        expires_at = key_data.get("expiresAt")
        if expires_at:
            # Handle Firebase timestamp
            if hasattr(expires_at, "timestamp"):
                expiry_timestamp = expires_at.timestamp()
            else:
                expiry_timestamp = expires_at
                
            if expiry_timestamp < time.time():
                return False
        
        # Update usage data
        key_doc.reference.update({
            "lastUsed": firestore.SERVER_TIMESTAMP,
            "usageCount": firestore.Increment(1)
        })
        
        # Check rate limits
        if not check_rate_limit(api_key):
            print(f"API key {api_key} has exceeded its rate limit")
            return False
        
        return True
        
    except Exception as e:
        print(f"API key validation error: {e}")
        # For debugging, print stack trace
        import traceback
        traceback.print_exc()
        return False

def check_rate_limit(api_key):
    """
    Check if an API key is within its rate limits.
    
    Args:
        api_key: The API key to check
        
    Returns:
        bool: True if within limits, False if rate limited
    """
    if not db or not api_key:
        return True  # Skip rate limiting if Firebase not available
    
    try:
        # Get the API key document
        print(f"Checking rate limit for key: {api_key}")
        query = db.collection("api_keys").where("key", "==", api_key).limit(1)
        results = list(query.stream())
        
        if not results:
            return False
        
        key_doc = results[0]
        key_data = key_doc.to_dict()
        print(f"Key document data: {key_data}")
        
        # Get the subscription tier
        tier = key_data.get("tier", "free")
        
        # Get or create usage tracking document
        usage_ref = db.collection("api_usage").document(key_doc.id)
        usage_doc = usage_ref.get()
        print(f"Usage document exists: {usage_doc.exists}")
        
        if usage_doc.exists:
            print(f"Usage data: {usage_doc.to_dict()}")
        
        current_time = int(time.time())
        day_start = current_time - (current_time % 86400)  # Start of current day
        month_start = current_time - (current_time % 2592000)  # Approximate start of month
        
        if not usage_doc.exists:
            # Initialize usage tracking
            usage_ref.set({
                "daily": {
                    "count": 1,
                    "reset_time": day_start
                },
                "monthly": {
                    "count": 1,
                    "reset_time": month_start
                },
                "last_updated": firestore.SERVER_TIMESTAMP
            })
            return True  # First request, always allowed
        
        # Get current usage data
        usage_data = usage_doc.to_dict()
        
        # Check if we need to reset daily counter
        daily = usage_data.get("daily", {})
        if daily.get("reset_time", 0) < day_start:
            daily = {"count": 0, "reset_time": day_start}
        
        # Check if we need to reset monthly counter
        monthly = usage_data.get("monthly", {})
        if monthly.get("reset_time", 0) < month_start:
            monthly = {"count": 0, "reset_time": month_start}
        
        # Define rate limits based on tier
        daily_limit = 100  # Default free tier
        monthly_limit = 3000
        
        if tier == "basic":
            daily_limit = 1000
            monthly_limit = 30000
        elif tier == "pro":
            daily_limit = 10000
            monthly_limit = 300000
        
        # Check if limits are exceeded
        if daily.get("count", 0) >= daily_limit:
            print(f"Rate limit exceeded: {daily.get('count')} requests > {daily_limit} daily limit")
            return False
        
        if monthly.get("count", 0) >= monthly_limit:
            print(f"Rate limit exceeded: {monthly.get('count')} requests > {monthly_limit} monthly limit")
            return False
        
        # Increment counters
        daily["count"] = daily.get("count", 0) + 1
        monthly["count"] = monthly.get("count", 0) + 1
        
        # Update usage tracking
        # Update usage tracking
        update_result = usage_ref.update({
            "daily": daily,
            "monthly": monthly,
            "last_updated": firestore.SERVER_TIMESTAMP
        })

        print(f"Updated usage tracking: {update_result}")
        print(f"Rate limit status: {daily['count']}/{daily_limit} daily, {monthly['count']}/{monthly_limit} monthly")
        return True
        
    except Exception as e:
        print(f"Rate limit check error: {e}")
        import traceback
        traceback.print_exc()
        return True  # On error, allow the request

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