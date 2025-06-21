"""
API helpers for Facebook Marketing API with rate limiting and error handling.
"""
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
from functools import wraps
import sqlite3
import os

logger = logging.getLogger(__name__)

# Rate limiting and caching configuration
API_CALL_COUNT = 0
SESSION_START = datetime.now()
CACHE_DB_PATH = "api_cache.db"

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
    param_str = json.dumps(params, sort_keys=True)
    return f"{endpoint}::{param_str}"

def get_cached_data(cache_key: str) -> Optional[Dict]:
    """Retrieve cached data if not expired."""
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
                return json.loads(data_str)
            else:
                # Clean up expired cache
                conn = sqlite3.connect(CACHE_DB_PATH)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM api_cache WHERE cache_key = ?", (cache_key,))
                conn.commit()
                conn.close()
    except Exception as e:
        logger.warning(f"Cache retrieval error: {e}")

    return None

def set_cached_data(cache_key: str, data: Dict, ttl_hours: int = 1):
    """Store data in cache with TTL."""
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        cursor.execute('''
            INSERT OR REPLACE INTO api_cache (cache_key, data, expires_at, created_at)
            VALUES (?, ?, ?, ?)
        ''', (
            cache_key,
            json.dumps(data),
            expires_at.isoformat(),
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Cache storage error: {e}")

def rate_limiter(calls_per_minute: int = 200):
    """Decorator to limit API calls per minute."""
    call_times = []

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove calls older than 1 minute
            call_times[:] = [t for t in call_times if now - t < 60]

            if len(call_times) >= calls_per_minute:
                sleep_time = 60 - (now - call_times[0])
                if sleep_time > 0:
                    logger.info(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                    time.sleep(sleep_time)

            call_times.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limiter(calls_per_minute=180)  # Conservative limit
def safe_api_call(
    api_func: Callable,
    endpoint: str,
    params: Dict = None,
    use_cache: bool = True,
    cache_ttl_hours: int = 1,
    force_refresh: bool = False,
    max_retries: int = 3
) -> Optional[Any]:
    """
    Safe wrapper for Facebook API calls with exponential backoff and caching.

    Args:
        api_func: The API function to call
        endpoint: API endpoint identifier for logging/caching
        params: API parameters
        use_cache: Whether to use caching
        cache_ttl_hours: Cache TTL in hours
        force_refresh: Force refresh cache
        max_retries: Maximum retry attempts

    Returns:
        API response data or None if failed
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

            # Cache the result
            if use_cache and result is not None:
                # Convert result to dict if it's a Facebook SDK object
                if hasattr(result, 'export_all_data'):
                    data_to_cache = [item.export_all_data() for item in result]
                elif hasattr(result, '__iter__') and not isinstance(result, (str, dict)):
                    data_to_cache = [
                        item.export_all_data() if hasattr(item, 'export_all_data') else item
                        for item in result
                    ]
                else:
                    data_to_cache = result

                set_cached_data(cache_key, data_to_cache, cache_ttl_hours)

            return result

        except Exception as e:
            error_msg = str(e).lower()

            # Check for rate limiting (code 17 or 4, throttling messages)
            is_rate_limit = (
                'code 17' in error_msg or 
                'code 4' in error_msg or
                'user request limit reached' in error_msg or
                'rate limit' in error_msg or
                'throttle' in error_msg or
                'too many requests' in error_msg
            )

            if is_rate_limit:
                sleep_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
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

def get_api_stats() -> Dict[str, Any]:
    """Get current session API usage statistics."""
    session_duration = (datetime.now() - SESSION_START).total_seconds()
    return {
        'total_calls': API_CALL_COUNT,
        'session_duration_minutes': session_duration / 60,
        'calls_per_minute': API_CALL_COUNT / (session_duration / 60) if session_duration > 0 else 0,
        'session_start': SESSION_START.isoformat()
    }

def batch_facebook_requests(requests: list, batch_size: int = 50) -> list:
    """
    Process Facebook API requests in batches.

    Args:
        requests: List of request objects
        batch_size: Maximum requests per batch

    Returns:
        List of responses
    """
    from facebook_business.api import FacebookAdsApi

    results = []

    for i in range(0, len(requests), batch_size):
        batch = requests[i:i + batch_size]

        try:
            logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} requests")
            batch_responses = safe_api_call(
                lambda: FacebookAdsApi.get_default_api().call_multiple(batch),
                f"batch_request_{i//batch_size}",
                {'batch_size': len(batch)},
                use_cache=False  # Batch requests are typically not cached
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
init_cache_db()
```

The code implements API helpers for Facebook Marketing API with rate limiting, error handling, and caching.
```python
"""
API helpers for Facebook Marketing API with rate limiting and error handling.
"""
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
from functools import wraps
import sqlite3
import os

logger = logging.getLogger(__name__)

# Rate limiting and caching configuration
API_CALL_COUNT = 0
SESSION_START = datetime.now()
CACHE_DB_PATH = "api_cache.db"

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
    param_str = json.dumps(params, sort_keys=True)
    return f"{endpoint}::{param_str}"

def get_cached_data(cache_key: str) -> Optional[Dict]:
    """Retrieve cached data if not expired."""
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
                return json.loads(data_str)
            else:
                # Clean up expired cache
                conn = sqlite3.connect(CACHE_DB_PATH)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM api_cache WHERE cache_key = ?", (cache_key,))
                conn.commit()
                conn.close()
    except Exception as e:
        logger.warning(f"Cache retrieval error: {e}")

    return None

def set_cached_data(cache_key: str, data: Dict, ttl_hours: int = 1):
    """Store data in cache with TTL."""
    try:
        conn = sqlite3.connect(CACHE_DB_PATH)
        cursor = conn.cursor()
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        cursor.execute('''
            INSERT OR REPLACE INTO api_cache (cache_key, data, expires_at, created_at)
            VALUES (?, ?, ?, ?)
        ''', (
            cache_key,
            json.dumps(data),
            expires_at.isoformat(),
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Cache storage error: {e}")

def rate_limiter(calls_per_minute: int = 200):
    """Decorator to limit API calls per minute."""
    call_times = []

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove calls older than 1 minute
            call_times[:] = [t for t in call_times if now - t < 60]

            if len(call_times) >= calls_per_minute:
                sleep_time = 60 - (now - call_times[0])
                if sleep_time > 0:
                    logger.info(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                    time.sleep(sleep_time)

            call_times.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limiter(calls_per_minute=180)  # Conservative limit
def safe_api_call(
    api_func: Callable,
    endpoint: str,
    params: Dict = None,
    use_cache: bool = True,
    cache_ttl_hours: int = 1,
    force_refresh: bool = False,
    max_retries: int = 3
) -> Optional[Any]:
    """
    Safe wrapper for Facebook API calls with exponential backoff and caching.

    Args:
        api_func: The API function to call
        endpoint: API endpoint identifier for logging/caching
        params: API parameters
        use_cache: Whether to use caching
        cache_ttl_hours: Cache TTL in hours
        force_refresh: Force refresh cache
        max_retries: Maximum retry attempts

    Returns:
        API response data or None if failed
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
            
            # Check rate limiting headers if available
            # Check rate limiting headers if available
            if hasattr(result, 'headers'):
                # Monitor app-level usage
                usage_header = result.headers.get('x-app-usage', '')
                if usage_header:
                    try:
                        import json
                        usage_data = json.loads(usage_header)
                        call_count = usage_data.get('call_count', 0)
                        total_time = usage_data.get('total_time', 0)
                        if call_count > 80:  # Approaching 100% limit
                            logger.warning(f"⚠️ High API usage: {call_count}% calls, {total_time}% time")
                            time.sleep(2)  # Proactive slowdown
                    except Exception as e:
                        logger.debug(f"Could not parse usage header: {e}")

                # Monitor ad account level usage
                ad_usage_header = result.headers.get('x-ad-account-usage', '')
                if ad_usage_header:
                    try:
                        import json
                        ad_usage_data = json.loads(ad_usage_header)
                        acc_id_usage = ad_usage_data.get('acc_id_util_pct', 0)
                        if acc_id_usage > 75:  # High ad account usage
                            logger.warning(f"⚠️ High ad account usage: {acc_id_usage}%")
                            time.sleep(1)
                    except Exception as e:
                        logger.debug(f"Could not parse ad usage header: {e}")

            # Cache the result
            if use_cache and result is not None:
                # Convert result to dict if it's a Facebook SDK object
                if hasattr(result, 'export_all_data'):
                    data_to_cache = [item.export_all_data() for item in result]
                elif hasattr(result, '__iter__') and not isinstance(result, (str, dict)):
                    data_to_cache = [
                        item.export_all_data() if hasattr(item, 'export_all_data') else item
                        for item in result
                    ]
                else:
                    data_to_cache = result

                set_cached_data(cache_key, data_to_cache, cache_ttl_hours)

            return result

        except Exception as e:
            error_msg = str(e).lower()

            # Check for rate limiting (code 17 or 4, throttling messages)
            is_rate_limit = (
                'code 17' in error_msg or 
                'code 4' in error_msg or
                'user request limit reached' in error_msg or
                'rate limit' in error_msg or
                'throttle' in error_msg or
                'too many requests' in error_msg
            )

            if is_rate_limit:
                sleep_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
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

def get_api_stats() -> Dict[str, Any]:
    """Get current session API usage statistics."""
    session_duration = (datetime.now() - SESSION_START).total_seconds()
    return {
        'total_calls': API_CALL_COUNT,
        'session_duration_minutes': session_duration / 60,
        'calls_per_minute': API_CALL_COUNT / (session_duration / 60) if session_duration > 0 else 0,
        'session_start': SESSION_START.isoformat()
    }

def batch_facebook_requests(requests: list, batch_size: int = 50) -> list:
    """
    Process Facebook API requests in batches.

    Args:
        requests: List of request objects
        batch_size: Maximum requests per batch

    Returns:
        List of responses
    """
    from facebook_business.api import FacebookAdsApi

    results = []

    for i in range(0, len(requests), batch_size):
        batch = requests[i:i + batch_size]

        try:
            logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} requests")
            batch_responses = safe_api_call(
                lambda: FacebookAdsApi.get_default_api().call_multiple(batch),
                f"batch_request_{i//batch_size}",
                {'batch_size': len(batch)},
                use_cache=False  # Batch requests are typically not cached
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
init_cache_db()
</replit_final_file>