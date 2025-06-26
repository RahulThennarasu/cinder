# Modified version of backend/auth/auth.py with automatic .env loading

import os
import hashlib
import hmac
import time
import logging
from typing import Optional, Dict, Any, List
import json

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Look for .env file in multiple locations
    env_paths = [
        os.path.join(os.path.dirname(__file__), '..', '..', '.env'),  # Root of project
        os.path.join(os.path.dirname(__file__), '..', '.env'),       # Backend folder
        os.path.join(os.path.dirname(__file__), '.env'),             # Auth folder
        '.env'  # Current working directory
    ]
    
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            logging.info(f"Loaded environment variables from {env_path}")
            break
    else:
        logging.info("No .env file found, using system environment variables")
        
except ImportError:
    logging.warning("python-dotenv not installed. Using system environment variables only.")

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    HAS_FIREBASE = True
except ImportError:
    HAS_FIREBASE = False
    logging.warning("Firebase not available. Install firebase-admin for full functionality.")

# Path to the file storing valid keys (fallback)
KEYS_FILE = os.path.join(os.path.dirname(__file__), 'valid_keys.json')

# In-memory cache of valid API keys
_API_KEY_CACHE: Dict[str, Dict[str, Any]] = {}

# Master key for admin/development use
_MASTER_KEY = os.environ.get("CINDER_MASTER_KEY", "cinder_master_key_change_me")

# Initialize Firebase using environment variables
db = None
if HAS_FIREBASE:
    try:
        # Get Firebase configuration from environment variables
        firebase_config = {
            "type": "service_account",
            "project_id": os.environ.get("FIREBASE_PROJECT_ID"),
            "private_key_id": os.environ.get("FIREBASE_PRIVATE_KEY_ID"),
            "private_key": os.environ.get("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n'),
            "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL"),
            "client_id": os.environ.get("FIREBASE_CLIENT_ID"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.environ.get("FIREBASE_CLIENT_CERT_URL"),
            "universe_domain": "googleapis.com"
        }
        
        # Check if all required environment variables are present
        required_vars = ["FIREBASE_PROJECT_ID", "FIREBASE_PRIVATE_KEY", "FIREBASE_CLIENT_EMAIL"]
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            logging.warning(f"Missing Firebase environment variables: {missing_vars}")
            logging.info("Firebase authentication disabled. Using local fallback.")
        else:
            # Check if Firebase is already initialized
            if not firebase_admin._apps:
                cred = credentials.Certificate(firebase_config)
                firebase_admin.initialize_app(cred)
                db = firestore.client()
                logging.info("Firebase initialized successfully using environment variables")
            else:
                db = firestore.client()
                
    except Exception as e:
        logging.error(f"Firebase initialization error: {e}")
        logging.info("Falling back to local authentication")
        db = None

def validate_api_key(api_key):
    """Validate an API key against Firebase database with local fallback."""
    if not api_key:
        return False
    
    # Check master key first
    if api_key == _MASTER_KEY:
        logging.info("Master key validated")
        return True
    
    # Try Firebase first if available
    if db and HAS_FIREBASE:
        try:
            return _validate_firebase_key(api_key)
        except Exception as e:
            logging.error(f"Firebase validation error: {e}")
            # Fall back to local validation
            return _validate_local_key(api_key)
    else:
        # Use local validation as fallback
        logging.info("Using local API key validation")
        return _validate_local_key(api_key)
def get_usage_stats_for_key(api_key: str) -> Optional[Dict[str, Any]]:
    """Get detailed usage statistics for an API key."""
    try:
        if not db or not HAS_FIREBASE:
            return None
        
        # Get the API key document
        query = db.collection("api_keys").where("key", "==", api_key).limit(1)
        results = list(query.stream())
        
        if not results:
            return None
        
        key_doc = results[0]
        key_data = key_doc.to_dict()
        tier = key_data.get("tier", "free")
        
        # Get usage document
        usage_ref = db.collection("api_usage").document(key_doc.id)
        usage_doc = usage_ref.get()
        
        current_time = int(time.time())
        day_start = current_time - (current_time % 86400)
        month_start = current_time - (current_time % 2592000)
        
        daily_usage = 0
        monthly_usage = 0
        
        if usage_doc.exists:
            usage_data = usage_doc.to_dict()
            
            # Get daily usage
            daily = usage_data.get("daily", {})
            if daily.get("reset_time", 0) >= day_start:
                daily_usage = daily.get("count", 0)
            
            # Get monthly usage
            monthly = usage_data.get("monthly", {})
            if monthly.get("reset_time", 0) >= month_start:
                monthly_usage = monthly.get("count", 0)
        
        # Define limits based on tier
        daily_limit = 100 if tier == "free" else 1000 if tier == "basic" else 10000
        monthly_limit = 3000 if tier == "free" else 30000 if tier == "basic" else 300000
        
        # Calculate reset times
        next_day = datetime.fromtimestamp(day_start + 86400)
        next_month_start = datetime.fromtimestamp(month_start + 2592000)
        
        # Get last used time
        last_used = key_data.get("lastUsed")
        last_used_str = None
        if last_used:
            if hasattr(last_used, "timestamp"):
                last_used_str = datetime.fromtimestamp(last_used.timestamp()).isoformat()
            else:
                last_used_str = last_used
        
        return {
            "api_key_id": key_doc.id,
            "daily_usage": daily_usage,
            "daily_limit": daily_limit,
            "monthly_usage": monthly_usage,
            "monthly_limit": monthly_limit,
            "last_used": last_used_str,
            "total_requests": key_data.get("usageCount", 0),
            "tier": tier,
            "reset_times": {
                "daily_reset": next_day.isoformat(),
                "monthly_reset": next_month_start.isoformat()
            }
        }
        
    except Exception as e:
        logging.error(f"Error getting usage stats: {e}")
        return None

def get_usage_history_for_key(api_key: str, days: int = 30) -> List[Dict[str, Any]]:
    """Get usage history for an API key over the last N days."""
    try:
        if not db or not HAS_FIREBASE:
            return []
        
        # Get the API key document
        query = db.collection("api_keys").where("key", "==", api_key).limit(1)
        results = list(query.stream())
        
        if not results:
            return []
        
        key_doc = results[0]
        
        # Get usage history from the usage_history subcollection
        history_ref = db.collection("api_usage").document(key_doc.id).collection("daily_history")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Query for usage history
        history_query = history_ref.where("date", ">=", start_date.strftime("%Y-%m-%d")).order_by("date", direction=firestore.Query.DESCENDING)
        history_docs = list(history_query.stream())
        
        # Convert to list format
        history = []
        for doc in history_docs:
            data = doc.to_dict()
            history.append({
                "date": data.get("date"),
                "requests": data.get("requests", 0)
            })
        
        # Fill in missing days with 0 requests
        history_dict = {item["date"]: item["requests"] for item in history}
        
        complete_history = []
        for i in range(days):
            date = (end_date - timedelta(days=i)).strftime("%Y-%m-%d")
            complete_history.append({
                "date": date,
                "requests": history_dict.get(date, 0)
            })
        
        return complete_history
        
    except Exception as e:
        logging.error(f"Error getting usage history: {e}")
        return []

def log_api_usage(api_key: str, endpoint: str = "unknown"):
    """Log API usage for analytics (call this from your API endpoints)."""
    try:
        if not db or not HAS_FIREBASE:
            return
        
        # Get the API key document
        query = db.collection("api_keys").where("key", "==", api_key).limit(1)
        results = list(query.stream())
        
        if not results:
            return
        
        key_doc = results[0]
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Update daily history
        history_ref = db.collection("api_usage").document(key_doc.id).collection("daily_history").document(today)
        
        # Use a transaction to safely increment the counter
        @firestore.transactional
        def update_daily_history(transaction):
            doc = history_ref.get(transaction=transaction)
            if doc.exists:
                transaction.update(history_ref, {
                    "requests": firestore.Increment(1),
                    "last_endpoint": endpoint,
                    "last_used": firestore.SERVER_TIMESTAMP
                })
            else:
                transaction.set(history_ref, {
                    "date": today,
                    "requests": 1,
                    "last_endpoint": endpoint,
                    "last_used": firestore.SERVER_TIMESTAMP
                })
        
        # Execute the transaction
        transaction = db.transaction()
        update_daily_history(transaction)
        
    except Exception as e:
        logging.error(f"Error logging API usage: {e}")

# Enhanced rate limiting with better tracking
def _check_firebase_rate_limit(api_key):
    """Enhanced rate limit checking with usage logging."""
    try:
        # Get the API key document
        query = db.collection("api_keys").where("key", "==", api_key).limit(1)
        results = list(query.stream())
        
        if not results:
            return False
        
        key_doc = results[0]
        key_data = key_doc.to_dict()
        
        # Get the subscription tier
        tier = key_data.get("tier", "free")
        
        # Get or create usage tracking document
        usage_ref = db.collection("api_usage").document(key_doc.id)
        usage_doc = usage_ref.get()
        
        current_time = int(time.time())
        day_start = current_time - (current_time % 86400)
        month_start = current_time - (current_time % 2592000)
        
        if not usage_doc.exists:
            # Initialize usage tracking
            usage_ref.set({
                "daily": {"count": 1, "reset_time": day_start},
                "monthly": {"count": 1, "reset_time": month_start},
                "last_updated": firestore.SERVER_TIMESTAMP,
                "created_at": firestore.SERVER_TIMESTAMP
            })
            
            # Also log the usage
            log_api_usage(api_key, "api_call")
            return True
        
        # Get current usage data
        usage_data = usage_doc.to_dict()
        
        # Check if we need to reset counters
        daily = usage_data.get("daily", {})
        if daily.get("reset_time", 0) < day_start:
            daily = {"count": 0, "reset_time": day_start}
        
        monthly = usage_data.get("monthly", {})
        if monthly.get("reset_time", 0) < month_start:
            monthly = {"count": 0, "reset_time": month_start}
        
        # Define rate limits based on tier
        daily_limit = 100 if tier == "free" else 1000 if tier == "basic" else 10000
        monthly_limit = 3000 if tier == "free" else 30000 if tier == "basic" else 300000
        
        # Check if limits are exceeded
        if daily.get("count", 0) >= daily_limit:
            logging.warning(f"Daily rate limit exceeded for {api_key[:10]}... ({daily.get('count', 0)}/{daily_limit})")
            return False
        
        if monthly.get("count", 0) >= monthly_limit:
            logging.warning(f"Monthly rate limit exceeded for {api_key[:10]}... ({monthly.get('count', 0)}/{monthly_limit})")
            return False
        
        # Increment counters
        daily["count"] = daily.get("count", 0) + 1
        monthly["count"] = monthly.get("count", 0) + 1
        
        # Update usage tracking
        usage_ref.update({
            "daily": daily,
            "monthly": monthly,
            "last_updated": firestore.SERVER_TIMESTAMP
        })
        
        # Log the usage for analytics
        log_api_usage(api_key, "api_call")
        
        logging.info(f"API usage logged: {daily['count']}/{daily_limit} daily, {monthly['count']}/{monthly_limit} monthly")
        return True
        
    except Exception as e:
        logging.error(f"Firebase rate limit check error: {e}")
        return True
    
def _validate_firebase_key(api_key):
    """Validate API key using Firebase."""
    try:
        # Query Firestore for the API key
        query = db.collection("api_keys").where("key", "==", api_key).limit(1)
        results = list(query.stream())
        
        if not results:
            logging.info(f"API key not found in Firebase: {api_key[:10]}...")
            return False
        
        key_doc = results[0]
        key_data = key_doc.to_dict()
        
        # Check if the key is active
        if not key_data.get("active", True):
            logging.info(f"API key is inactive: {api_key[:10]}...")
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
                logging.info(f"API key expired: {api_key[:10]}...")
                return False
        
        # Update usage data
        try:
            key_doc.reference.update({
                "lastUsed": firestore.SERVER_TIMESTAMP,
                "usageCount": firestore.Increment(1)
            })
        except Exception as e:
            logging.warning(f"Could not update usage data: {e}")
        
        # Cache the key data
        _API_KEY_CACHE[api_key] = key_data
        
        logging.info(f"API key validated successfully via Firebase: {api_key[:10]}...")
        return True
        
    except Exception as e:
        logging.error(f"Firebase API key validation error: {e}")
        raise

def _validate_local_key(api_key):
    """Validate API key using local JSON file (fallback)."""
    try:
        # Check if it looks like a valid Cinder API key format
        if not api_key.startswith("cinder_"):
            logging.info(f"Invalid API key format: {api_key[:10]}...")
            return False
        
        # For development/testing: if the key format is correct, allow it
        # This is a temporary measure while Firebase is being set up
        if len(api_key) > 40 and "_" in api_key:
            logging.info(f"API key format validated (development mode): {api_key[:10]}...")
            return True
        
        # Check local storage
        valid_keys = _load_valid_keys()
        
        if api_key not in valid_keys:
            logging.info(f"API key not found locally: {api_key[:10]}...")
            return False
        
        key_data = valid_keys[api_key]
        
        # Check if expired
        expires_at = key_data.get("expires_at", 0)
        if expires_at and expires_at < time.time():
            logging.info(f"Local API key expired: {api_key[:10]}...")
            return False
        
        # Cache the key data
        _API_KEY_CACHE[api_key] = key_data
        logging.info(f"API key validated successfully via local storage: {api_key[:10]}...")
        return True
        
    except Exception as e:
        logging.error(f"Local key validation error: {e}")
        return False

def check_rate_limit(api_key):
    """Check if an API key is within its rate limits."""
    if not api_key or api_key == _MASTER_KEY:
        return True  # Master key has no limits
    
    # Try Firebase first if available
    if db and HAS_FIREBASE:
        try:
            return _check_firebase_rate_limit(api_key)
        except Exception as e:
            logging.error(f"Firebase rate limit check error: {e}")
            return True  # Allow on error for now
    else:
        # Local rate limiting is basic - just return True for development
        logging.info("Using local rate limiting (no limits)")
        return True

def _check_firebase_rate_limit(api_key):
    """Check rate limits using Firebase."""
    try:
        # Get the API key document
        query = db.collection("api_keys").where("key", "==", api_key).limit(1)
        results = list(query.stream())
        
        if not results:
            return False
        
        key_doc = results[0]
        key_data = key_doc.to_dict()
        
        # Get the subscription tier
        tier = key_data.get("tier", "free")
        
        # Get or create usage tracking document
        usage_ref = db.collection("api_usage").document(key_doc.id)
        usage_doc = usage_ref.get()
        
        current_time = int(time.time())
        day_start = current_time - (current_time % 86400)
        month_start = current_time - (current_time % 2592000)
        
        if not usage_doc.exists:
            # Initialize usage tracking
            usage_ref.set({
                "daily": {"count": 1, "reset_time": day_start},
                "monthly": {"count": 1, "reset_time": month_start},
                "last_updated": firestore.SERVER_TIMESTAMP
            })
            return True
        
        # Get current usage data
        usage_data = usage_doc.to_dict()
        
        # Check if we need to reset counters
        daily = usage_data.get("daily", {})
        if daily.get("reset_time", 0) < day_start:
            daily = {"count": 0, "reset_time": day_start}
        
        monthly = usage_data.get("monthly", {})
        if monthly.get("reset_time", 0) < month_start:
            monthly = {"count": 0, "reset_time": month_start}
        
        # Define rate limits based on tier
        daily_limit = 100 if tier == "free" else 1000 if tier == "basic" else 10000
        monthly_limit = 3000 if tier == "free" else 30000 if tier == "basic" else 300000
        
        # Check if limits are exceeded
        if daily.get("count", 0) >= daily_limit:
            logging.warning(f"Daily rate limit exceeded for {api_key[:10]}...")
            return False
        
        if monthly.get("count", 0) >= monthly_limit:
            logging.warning(f"Monthly rate limit exceeded for {api_key[:10]}...")
            return False
        
        # Increment counters
        daily["count"] = daily.get("count", 0) + 1
        monthly["count"] = monthly.get("count", 0) + 1
        
        # Update usage tracking
        usage_ref.update({
            "daily": daily,
            "monthly": monthly,
            "last_updated": firestore.SERVER_TIMESTAMP
        })
        
        return True
        
    except Exception as e:
        logging.error(f"Firebase rate limit check error: {e}")
        return True

def generate_api_key(user_id: str, tier: str = "free") -> str:
    """Generate a new API key and add it to Firebase or local storage."""
    timestamp = str(int(time.time()))
    
    # Create a unique key
    secret = os.environ.get("CINDER_API_SECRET", "change_this_secret_key")
    message = f"{user_id}:{timestamp}:{tier}"
    
    # Generate HMAC using SHA-256
    signature = hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Create the formatted API key
    api_key = f"cinder_{timestamp}_{signature[:32]}"
    
    # Try to add to Firebase first
    if db and HAS_FIREBASE:
        try:
            from datetime import datetime
            # Add to Firebase
            doc_ref = db.collection("api_keys").add({
                "key": api_key,
                "userId": user_id,
                "tier": tier,
                "createdAt": firestore.SERVER_TIMESTAMP,
                "expiresAt": firestore.Timestamp.from_datetime(
                    datetime.fromtimestamp(time.time() + 365 * 24 * 60 * 60)
                ),
                "lastUsed": None,
                "usageCount": 0,
                "active": True
            })
            logging.info(f"API key added to Firebase: {api_key[:10]}...")
            
        except Exception as e:
            logging.error(f"Error adding key to Firebase: {e}")
            # Fall back to local storage
            _add_key_locally(api_key, user_id, tier)
    else:
        # Use local storage
        _add_key_locally(api_key, user_id, tier)
    
    return api_key

def _add_key_locally(api_key: str, user_id: str, tier: str):
    """Add API key to local JSON file."""
    valid_keys = _load_valid_keys()
    valid_keys[api_key] = {
        "user_id": user_id,
        "tier": tier,
        "created_at": time.time(),
        "expires_at": time.time() + (365 * 24 * 60 * 60),
        "permissions": ["basic"]
    }
    _save_valid_keys(valid_keys)
    logging.info(f"API key added locally: {api_key[:10]}...")

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
        os.makedirs(os.path.dirname(KEYS_FILE), exist_ok=True)
        with open(KEYS_FILE, 'w') as f:
            json.dump(keys, f, indent=2)
        return True
    except Exception as e:
        logging.error(f"Error saving keys file: {e}")
        return False

def get_key_permissions(api_key: Optional[str]) -> List[str]:
    """Get the permissions associated with an API key."""
    if not api_key:
        return []
    
    if api_key in _API_KEY_CACHE:
        return _API_KEY_CACHE[api_key].get("permissions", ["basic"])
    
    if validate_api_key(api_key):
        return _API_KEY_CACHE[api_key].get("permissions", ["basic"])
    
    return []

def revoke_api_key(api_key: str) -> bool:
    """Revoke an API key."""
    # Remove from cache
    if api_key in _API_KEY_CACHE:
        del _API_KEY_CACHE[api_key]
    
    # Try Firebase first
    if db and HAS_FIREBASE:
        try:
            query = db.collection("api_keys").where("key", "==", api_key).limit(1)
            results = list(query.stream())
            
            if results:
                results[0].reference.update({"active": False})
                return True
        except Exception as e:
            logging.error(f"Error revoking Firebase key: {e}")
    
    # Fall back to local storage
    valid_keys = _load_valid_keys()
    if api_key in valid_keys:
        del valid_keys[api_key]
        return _save_valid_keys(valid_keys)
    
    return False

def list_valid_keys() -> List[Dict[str, Any]]:
    """List all valid API keys."""
    keys = []
    
    # Try Firebase first
    if db and HAS_FIREBASE:
        try:
            keys_ref = db.collection("api_keys").where("active", "==", True)
            for doc in keys_ref.stream():
                key_data = doc.to_dict()
                keys.append({
                    "key": f"{key_data.get('key', '')[:10]}...{key_data.get('key', '')[-5:]}",
                    "user_id": key_data.get("userId", "unknown"),
                    "tier": key_data.get("tier", "free"),
                    "created_at": key_data.get("createdAt", 0),
                    "expires_at": key_data.get("expiresAt", 0),
                    "permissions": ["basic"]
                })
            return keys
        except Exception as e:
            logging.error(f"Error listing Firebase keys: {e}")
    
    # Fall back to local storage
    valid_keys = _load_valid_keys()
    
    for key, info in valid_keys.items():
        masked_key = f"{key[:10]}...{key[-5:]}"
        keys.append({
            "key": masked_key,
            "user_id": info.get("user_id", "unknown"),
            "tier": info.get("tier", "free"),
            "created_at": info.get("created_at", 0),
            "expires_at": info.get("expires_at", 0),
            "permissions": info.get("permissions", [])
        })
    
    return keys