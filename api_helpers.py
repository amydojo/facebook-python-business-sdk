
"""
api_helpers.py

Provides safe, rate-limited, cached wrappers around Meta Graph API calls for paid and organic data.
Implements exponential backoff, caching, pagination, and bulk-fetch utilities.
"""

import os
import time
import json
import logging
import threading
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Tuple, List, Union
from functools import wraps
import requests

# Import Facebook Business SDK if available
try:
    from facebook_business.api import FacebookAdsApi
    from facebook_business.exceptions import FacebookRequestError, FacebookError
    SDK_AVAILABLE = True
except ImportError:
    FacebookAdsApi = None
    FacebookRequestError = Exception
    FacebookError = Exception
    SDK_AVAILABLE = False

logger = logging.getLogger(__name__)

# Configuration via environment variables with defaults
API_VERSION = os.getenv("GRAPH_API_VERSION", "v21.0")
GRAPH_API_BASE = f"https://graph.facebook.com/{API_VERSION}"
MAX_RETRIES = int(os.getenv("API_HELPERS_MAX_RETRIES", "3"))
BACKOFF_BASE = float(os.getenv("API_HELPERS_BACKOFF_BASE", "2.0"))
BACKOFF_MAX = float(os.getenv("API_HELPERS_BACKOFF_MAX", "60.0"))
REQUEST_TIMEOUT = float(os.getenv("API_HELPERS_REQUEST_TIMEOUT", "15.0"))
CACHE_ENABLED = os.getenv("API_HELPERS_CACHE_ENABLED", "true").lower() in ("1", "true", "yes")
CACHE_EXPIRATION_SECONDS = int(os.getenv("API_HELPERS_CACHE_EXPIRATION_SECONDS", "3600"))

# Rate limiting configuration
RATE_LIMIT_CALLS_PER_MINUTE = int(os.getenv("API_HELPERS_RATE_LIMIT", "180"))

# SQLite cache database
CACHE_DB_PATH = "api_cache.db"

# Thread-safe rate limiting
_rate_limit_lock = threading.Lock()
_call_timestamps = []

# Session tracking
API_CALL_COUNT = 0
SESSION_START = datetime.now()

def init_cache_db():
    """Initialize SQLite cache database."""
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_cache (
            cache_key TEXT PRIMARY KEY,
            data TEXT,
            expires_at TEXT,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

def get_cache_key(endpoint: str, params: Dict) -> str:
    """Generate cache key from endpoint and parameters."""
    try:
        param_str = json.dumps(params, sort_keys=True, default=str)
        return f"{endpoint}::{param_str}"
    except Exception:
        return f"{endpoint}::{str(params)}"

def get_cached_data(cache_key: str) -> Optional[Dict]:
    """Retrieve cached data if not expired."""
    if not CACHE_ENABLED:
        return None
    
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT data, expires_at FROM api_cache WHERE cache_key = ?",
            (cache_key,)
        )
        result = cursor.fetchone()
        conn.close()

        if result:
            data_str, expires_at = result
            if datetime.fromisoformat(expires_at) > datetime.now():
                logger.debug(f"Cache HIT for {cache_key}")
                return json.loads(data_str)
            else:
                # Clean up expired cache
                conn = sqlite3.connect(CACHE_DB_PATH)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM api_cache WHERE cache_key = ?", (cache_key,))
                conn.commit()
                conn.close()
                logger.debug(f"Cache EXPIRED for {cache_key}")
    except Exception as e:
        logger.warning(f"Cache retrieval error: {e}")

    return None

def set_cached_data(cache_key: str, data: Any, ttl_hours: int = 1):
    """Store data in cache with TTL."""
    if not CACHE_ENABLED:
        return
    
    try:
        # Test JSON serialization
        json.dumps(data, default=str)
        
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        cursor.execute('''
            INSERT OR REPLACE INTO api_cache (cache_key, data, expires_at, created_at)
            VALUES (?, ?, ?, ?)
        ''', (
            cache_key,
            json.dumps(data, default=str),
            expires_at.isoformat(),
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
        logger.debug(f"Cache SET for {cache_key}")
    except Exception as e:
        logger.warning(f"Cache storage error: {e}")

def rate_limiter(calls_per_minute: int = RATE_LIMIT_CALLS_PER_MINUTE):
    """Decorator to limit API calls per minute."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with _rate_limit_lock:
                now = time.time()
                # Remove calls older than 1 minute
                _call_timestamps[:] = [t for t in _call_timestamps if now - t < 60]

                if len(_call_timestamps) >= calls_per_minute:
                    sleep_time = 60 - (now - _call_timestamps[0])
                    if sleep_time > 0:
                        logger.info(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                        time.sleep(sleep_time)

                _call_timestamps.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limiter()
def safe_api_call(
    api_func: Callable,
    endpoint: str,
    params: Dict = None,
    use_cache: bool = True,
    cache_ttl_hours: int = 1,
    force_refresh: bool = False,
    max_retries: int = MAX_RETRIES
) -> Optional[Any]:
    """
    Safe wrapper for API calls with exponential backoff and caching.

    Args:
        api_func: The API function to call
        endpoint: API endpoint identifier for logging/caching
        params: API parameters
        use_cache: Whether to use caching
        cache_ttl_hours: Cache TTL in hours
        force_refresh: Force refresh cache
        max_retries: Maximum retry attempts

    Returns:
        Parsed JSON data (dict/list) or None if failed
    """
    global API_CALL_COUNT

    if params is None:
        params = {}

    # Check cache first
    cache_key = get_cache_key(endpoint, params)
    if use_cache and not force_refresh:
        cached_data = get_cached_data(cache_key)
        if cached_data:
            logger.info(f"Cache hit for {endpoint}")
            return cached_data

    # Track API calls
    API_CALL_COUNT += 1

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            logger.info(f"API call #{API_CALL_COUNT} to {endpoint}, attempt {attempt + 1}")

            # Make the API call
            result = api_func()

            elapsed = time.time() - start_time
            logger.info(f"API call successful in {elapsed:.2f}s")

            # Parse response if it's an HTTP response object
            parsed_result = result
            if hasattr(result, 'json') and hasattr(result, 'status_code'):
                # It's an HTTP response - parse JSON and check status
                if result.status_code != 200:
                    try:
                        error_body = result.json()
                        error_msg = error_body.get('error', {}).get('message', f'HTTP {result.status_code}')
                        raise Exception(f"API error: {error_msg}")
                    except ValueError:
                        raise Exception(f"HTTP {result.status_code}: {result.text}")
                
                try:
                    parsed_result = result.json()
                except ValueError:
                    raise Exception("Invalid JSON response")

            # Cache the result
            if use_cache and parsed_result is not None:
                # Convert result to dict if it's a Facebook SDK object
                if hasattr(parsed_result, 'export_all_data'):
                    data_to_cache = [item.export_all_data() for item in parsed_result]
                elif hasattr(parsed_result, '__iter__') and not isinstance(parsed_result, (str, dict)):
                    try:
                        data_to_cache = [
                            item.export_all_data() if hasattr(item, 'export_all_data') else item
                            for item in parsed_result
                        ]
                    except:
                        data_to_cache = parsed_result
                else:
                    data_to_cache = parsed_result

                set_cached_data(cache_key, data_to_cache, cache_ttl_hours)

            return parsed_result

        except Exception as e:
            error_msg = str(e).lower()

            # Check for rate limiting
            is_rate_limit = (
                'code 17' in error_msg or 
                'code 4' in error_msg or
                'user request limit reached' in error_msg or
                'rate limit' in error_msg or
                'throttle' in error_msg or
                'too many requests' in error_msg
            )

            if is_rate_limit:
                sleep_time = (BACKOFF_BASE ** attempt) * 5  # Exponential backoff
                sleep_time = min(sleep_time, BACKOFF_MAX)
                logger.warning(f"Rate limit detected on attempt {attempt + 1}, sleeping {sleep_time}s")

                if attempt < max_retries - 1:
                    time.sleep(sleep_time)
                    continue
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} attempts for {endpoint}")
                    return None
            else:
                logger.error(f"API call failed for {endpoint}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Short delay for other errors
                    continue
                else:
                    return None

    return None

def http_get(url: str, params: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Simple GET via requests, wrapped in safe_api_call. Returns JSON dict or raises.
    """
    def _call():
        resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        try:
            body = resp.json()
        except ValueError:
            resp.raise_for_status()
            return {}
        
        if resp.status_code != 200:
            error_info = body.get("error") if isinstance(body, dict) else resp.text
            raise Exception(f"HTTP {resp.status_code}: {error_info}")
        return body
    
    return safe_api_call(_call, url, params)

def http_post(url: str, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Simple POST via requests, wrapped in safe_api_call. Returns JSON dict or raises.
    """
    def _call():
        resp = requests.post(url, data=data, headers=headers, timeout=REQUEST_TIMEOUT)
        try:
            body = resp.json()
        except ValueError:
            resp.raise_for_status()
            return {}
        
        if resp.status_code not in (200, 201):
            error_info = body.get("error") if isinstance(body, dict) else resp.text
            raise Exception(f"HTTP {resp.status_code}: {error_info}")
        return body
    
    return safe_api_call(_call, url, data)

def paginate_get(
    endpoint: str,
    params: Dict[str, Any],
    limit_per_page: int = 100,
    max_pages: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Automatically paginate through Graph API paged GET results.
    
    Args:
        endpoint: full URL
        params: dict of parameters
        limit_per_page: how many items per page
        max_pages: optional max number of pages to fetch
        
    Returns:
        List combining all data entries
    """
    all_data = []
    page_count = 0
    next_after = None
    
    while True:
        p = params.copy()
        p['limit'] = limit_per_page
        if next_after:
            p['after'] = next_after
            
        logger.debug(f"paginate_get: fetching page {page_count+1} from {endpoint}")
        resp = http_get(endpoint, p)
        
        if not resp:
            break
            
        data = resp.get('data', [])
        all_data.extend(data)
        page_count += 1
        
        # Break conditions
        paging = resp.get('paging', {})
        cursors = paging.get('cursors', {})
        next_after = cursors.get('after')
        
        if not next_after:
            break
        if max_pages and page_count >= max_pages:
            break
    
    logger.info(f"paginate_get: fetched {len(all_data)} items from {endpoint} after {page_count} pages")
    return all_data

def get_graph_api_url(path: str) -> str:
    """
    Build full Graph API URL for a given path.
    
    Args:
        path: API path (without leading slash)
        
    Returns:
        Full Graph API URL
    """
    path = path.lstrip('/')
    return f"{GRAPH_API_BASE}/{path}"

def fetch_insights_with_paging(
    object_id: str,
    metrics: List[str],
    since: Optional[str] = None,
    until: Optional[str] = None,
    period: str = "day",
    extra_params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Fetch insights for a Page or Instagram media with automatic pagination.
    
    Args:
        object_id: page_id or media_id
        metrics: list of metric names
        since/until: 'YYYY-MM-DD' strings
        period: e.g. 'day'
        extra_params: additional parameters
        
    Returns:
        List of raw insight objects
    """
    path = f"{object_id}/insights"
    url = get_graph_api_url(path)
    params: Dict[str, Any] = {"metric": ",".join(metrics), "period": period}
    
    token = os.getenv("PAGE_ACCESS_TOKEN") or os.getenv("META_ACCESS_TOKEN")
    if token:
        params["access_token"] = token
    if since:
        params["since"] = since
    if until:
        params["until"] = until
    if extra_params:
        params.update(extra_params)
    
    logger.info(f"fetch_insights_with_paging: {url}")
    try:
        data = paginate_get(url, params)
        return data
    except Exception as e:
        logger.error(f"fetch_insights_with_paging failed: {e}", exc_info=True)
        return []

def sdk_call_with_backoff(sdk_func: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Wrap SDK calls with safe_api_call for retry logic.
    
    Args:
        sdk_func: function from facebook_business SDK
        
    Returns:
        SDK response or None if failed
    """
    def _call():
        return sdk_func(*args, **kwargs)
    
    endpoint = f"sdk_{sdk_func.__name__}"
    return safe_api_call(_call, endpoint, {})

def fetch_ad_insights_fields(
    account_id: str,
    level: str,
    fields: List[str],
    date_preset: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    params_extra: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Fetch ad insights using facebook_business SDK or HTTP call.
    
    Args:
        account_id: Ad account ID
        level: 'campaign', 'adset', or 'ad'
        fields: list of insight fields
        date_preset: preset like 'last_7d'
        since/until: custom date range
        params_extra: additional parameters
        
    Returns:
        List of insight records
    """
    # Use HTTP call for more reliable results
    account_id = account_id if account_id.startswith("act_") else f"act_{account_id}"
    path = f"{account_id}/insights"
    url = get_graph_api_url(path)
    
    params: Dict[str, Any] = {
        "level": level,
        "fields": ",".join(fields)
    }
    
    if date_preset:
        params["date_preset"] = date_preset
    if since and until:
        params["time_range"] = json.dumps({"since": since, "until": until})
    if params_extra:
        params.update(params_extra)
    
    token = os.getenv("META_ACCESS_TOKEN")
    if token:
        params["access_token"] = token
    
    try:
        data = paginate_get(url, params)
        return data
    except Exception as e:
        logger.error(f"fetch_ad_insights_fields failed: {e}")
        return []

def validate_env_vars(vars_list: List[str]) -> Tuple[bool, List[str]]:
    """
    Check that each env var in vars_list is set and non-empty.
    
    Args:
        vars_list: list of environment variable names
        
    Returns:
        Tuple of (all_present, missing_list)
    """
    missing = [v for v in vars_list if not os.getenv(v)]
    return (len(missing) == 0, missing)

def get_api_stats() -> Dict[str, Any]:
    """Get current session API usage statistics."""
    session_duration = (datetime.now() - SESSION_START).total_seconds()
    return {
        'total_calls': API_CALL_COUNT,
        'session_duration_minutes': session_duration / 60,
        'calls_per_minute': API_CALL_COUNT / (session_duration / 60) if session_duration > 0 else 0,
        'session_start': SESSION_START.isoformat()
    }

def batch_facebook_requests(requests_list: list, batch_size: int = 50) -> list:
    """
    Process Facebook API requests in batches.

    Args:
        requests_list: List of request objects
        batch_size: Maximum requests per batch

    Returns:
        List of responses
    """
    if not SDK_AVAILABLE:
        logger.error("Facebook Business SDK not available for batch requests")
        return []

    results = []

    for i in range(0, len(requests_list), batch_size):
        batch = requests_list[i:i + batch_size]

        try:
            logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} requests")
            
            def batch_call():
                return FacebookAdsApi.get_default_api().call_multiple(batch)
            
            batch_responses = safe_api_call(
                batch_call,
                f"batch_request_{i//batch_size}",
                {'batch_size': len(batch)},
                use_cache=False
            )

            if batch_responses:
                results.extend(batch_responses)

            # Small delay between batches
            time.sleep(1)

        except Exception as e:
            logger.error(f"Batch request failed: {e}")
            continue

    return results

# Initialize cache on import
try:
    init_cache_db()
except Exception as e:
    logger.warning(f"Failed to initialize cache database: {e}")
