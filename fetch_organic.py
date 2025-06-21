
"""
Enhanced fetch organic content insights from Facebook Pages and Instagram.
Handles comprehensive Instagram Business Account insights with metadata-driven discovery.

Official docs:
- Page Insights: https://developers.facebook.com/docs/graph-api/reference/page/insights/
- Instagram Insights: https://developers.facebook.com/docs/instagram-api/guides/insights/
- Token requirements: https://developers.facebook.com/docs/facebook-login/access-tokens

Updated with comprehensive Instagram metrics discovery, advanced KPIs, and optimized performance
"""
import os
import requests
import pandas as pd
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Union, Tuple
from config import config

logger = logging.getLogger(__name__)

# Graph API version and base URL for consistent endpoint calls
GRAPH_API_VERSION = os.getenv("GRAPH_API_VERSION", "v21.0")  # Updated to latest
GRAPH_API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"

# Import optimized API helpers
from api_helpers import safe_api_call, get_api_stats

# Session-level caches for Instagram insights optimization
_media_metrics_cache = {}          # key: (media_type, media_product_type), value: [metrics]
_ig_user_followers = None          # cache follower count
_ig_media_metadata_cache = {}      # key: media_id, value: metadata dict (supported metrics)
_ig_user_insights_cache = {}       # cache for user-level insights

# Cache for page insights metadata with comprehensive fallback
_cached_page_metrics = None
_metadata_fetch_failed = False

# Comprehensive fallback Page metrics when metadata fetch fails
FALLBACK_PAGE_METRICS = [
    "page_impressions", 
    "page_engaged_users", 
    "page_reach",
    "page_impressions_organic",
    "page_impressions_paid",
    "page_post_engagements",
    "page_views_total",
    "page_fan_adds",
    "page_fan_removes"
]

# Updated Instagram metrics for 2025 - removing deprecated metrics
FALLBACK_IG_METRICS = {
    # Core engagement metrics (most reliable)
    "reach": ["REEL", "VIDEO", "IMAGE", "CAROUSEL_ALBUM"],
    "total_interactions": ["REEL", "VIDEO", "IMAGE", "CAROUSEL_ALBUM"],
    "comments": ["REEL", "VIDEO", "IMAGE", "CAROUSEL_ALBUM"],
    "shares": ["REEL", "VIDEO", "IMAGE", "CAROUSEL_ALBUM"],
    "saved": ["REEL", "VIDEO", "IMAGE", "CAROUSEL_ALBUM"],
    "profile_visits": ["REEL", "VIDEO", "IMAGE", "CAROUSEL_ALBUM"],
    "follows": ["REEL", "VIDEO", "IMAGE", "CAROUSEL_ALBUM"],
    
    # Video/Reels specific metrics (updated for 2025)
    "video_views": ["VIDEO"],
    "ig_reels_avg_watch_time": ["REEL"],
    "ig_reels_video_view_total_time": ["REEL"],
    "clips_replays_count": ["REEL"],
    "ig_reels_aggregated_all_plays_count": ["REEL"],
    
    # Navigation and interaction metrics
    "navigation": ["REEL", "VIDEO"],
    "website_clicks": ["REEL", "VIDEO", "IMAGE", "CAROUSEL_ALBUM"],
    "email_contacts": ["REEL", "VIDEO", "IMAGE", "CAROUSEL_ALBUM"],
    "phone_call_clicks": ["REEL", "VIDEO", "IMAGE", "CAROUSEL_ALBUM"],
}

def get_page_access_token():
    """
    Get PAGE_ACCESS_TOKEN from environment, fallback to META_ACCESS_TOKEN.
    Returns tuple: (token, token_source)
    """
    page_token = os.getenv('PAGE_ACCESS_TOKEN')
    if page_token:
        logger.info("Using PAGE_ACCESS_TOKEN for organic insights")
        return page_token, "PAGE_ACCESS_TOKEN"

    meta_token = os.getenv('META_ACCESS_TOKEN')
    if meta_token:
        logger.warning("PAGE_ACCESS_TOKEN not found, falling back to META_ACCESS_TOKEN")
        return meta_token, "META_ACCESS_TOKEN"

    logger.error("Neither PAGE_ACCESS_TOKEN nor META_ACCESS_TOKEN found")
    return None, None

def validate_organic_environment():
    """
    Validate required environment variables for organic insights.
    Returns dict with validation status.
    """
    page_token, token_source = get_page_access_token()
    page_id = os.getenv('PAGE_ID')
    ig_user_id = os.getenv('IG_USER_ID')

    validation = {
        'page_token_available': bool(page_token),
        'page_id_available': bool(page_id),
        'ig_user_id_available': bool(ig_user_id),
        'token_source': token_source,
        'page_insights_enabled': bool(page_token and page_id),
        'instagram_insights_enabled': bool(page_token and ig_user_id)
    }

    logger.info(f"Organic insights validation: {validation}")
    return validation

def get_ig_follower_count(ig_user_id: str) -> Optional[int]:
    """
    Fetch Instagram Business Account follower count with caching.

    Args:
        ig_user_id: Instagram Business User ID

    Returns:
        Follower count or None if fetch fails
    """
    global _ig_user_followers

    if _ig_user_followers is not None:
        return _ig_user_followers

    token, _ = get_page_access_token()
    if not ig_user_id or not token:
        logger.warning("Missing IG_USER_ID or token for follower count.")
        return None

    url = f"{GRAPH_API_BASE}/{ig_user_id}"
    params = {"fields": "followers_count", "access_token": token}

    try:
        logger.info(f"Fetching follower count from: {url}")
        resp = requests.get(url, params=params, timeout=10)
        body = resp.json() if resp.headers.get("Content-Type", "").startswith("application/json") else {}
        
        logger.info(f"Follower count API response: status={resp.status_code}, body={body}")

        if resp.status_code == 200 and "followers_count" in body:
            _ig_user_followers = body["followers_count"]
            logger.info(f"✅ Fetched IG follower count: {_ig_user_followers}")
            return _ig_user_followers
        else:
            logger.warning(f"❌ Failed to fetch follower count: {body}")

    except Exception as e:
        logger.error(f"Error fetching IG follower count: {e}", exc_info=True)

    return None

def fetch_ig_user_insights(ig_user_id: str, metrics: List[str] = None, period: str = "lifetime") -> Dict:
    """
    Fetch Instagram Business Account user-level insights.

    Args:
        ig_user_id: Instagram Business User ID
        metrics: List of user-level metrics to fetch
        period: Time period ('day', 'week', 'days_28', 'lifetime')

    Returns:
        Dict with user-level insights data
    """
    if not metrics:
        metrics = ["profile_views", "website_clicks", "email_contacts"]

    cache_key = f"{ig_user_id}_{period}_{'_'.join(metrics)}"
    if cache_key in _ig_user_insights_cache:
        return _ig_user_insights_cache[cache_key]

    token, _ = get_page_access_token()
    if not ig_user_id or not token:
        logger.warning("Missing IG_USER_ID or token for user insights.")
        return {}

    url = f"{GRAPH_API_BASE}/{ig_user_id}/insights"
    params = {
        "metric": ",".join(metrics),
        "period": period,
        "access_token": token
    }

    try:
        logger.info(f"Fetching user insights from: {url} with params: {params}")
        resp = requests.get(url, params=params, timeout=10)
        body = resp.json() if resp.headers.get("Content-Type", "").startswith("application/json") else {}
        
        logger.info(f"User insights API response: status={resp.status_code}, body={body}")

        if resp.status_code == 200 and "data" in body:
            insights = {}
            for metric_obj in body["data"]:
                metric_name = metric_obj.get("name")
                values = metric_obj.get("values", [])
                if values:
                    insights[metric_name] = values[-1].get("value", 0)

            _ig_user_insights_cache[cache_key] = insights
            logger.info(f"✅ Fetched IG user insights: {list(insights.keys())}")
            return insights
        else:
            logger.warning(f"❌ Failed to fetch user insights: {body}")

    except Exception as e:
        logger.error(f"Error fetching IG user insights: {e}", exc_info=True)

    return {}

def fetch_media_insights_metadata(media_id: str) -> List[str]:
    """
    Fetch available metrics metadata for a specific media item.

    Args:
        media_id: Instagram media ID

    Returns:
        List of supported metric names for this media
    """
    if media_id in _ig_media_metadata_cache:
        return _ig_media_metadata_cache[media_id]

    token, _ = get_page_access_token()
    if not token:
        logger.warning("Missing token for media metadata.")
        return []

    url = f"{GRAPH_API_BASE}/{media_id}/insights/metadata"

    try:
        logger.info(f"Fetching metadata from: {url}")
        resp = requests.get(url, params={"access_token": token}, timeout=10)
        body = resp.json() if resp.headers.get("Content-Type", "").startswith("application/json") else {}
        
        logger.info(f"Metadata API response: status={resp.status_code}, body={body}")

        if resp.status_code == 200 and "data" in body:
            metrics = [item.get("name") for item in body["data"] if item.get("name")]
            logger.info(f"📊 Media {media_id} supports metrics: {metrics}")
            _ig_media_metadata_cache[media_id] = metrics
            return metrics
        else:
            logger.info(f"⚠️ Metadata fetch not available for media {media_id}: {body}")

    except Exception as e:
        logger.debug(f"Metadata fetch failed for media {media_id}: {e}")

    _ig_media_metadata_cache[media_id] = []
    return []

def choose_metrics_for_media(media: Dict) -> List[str]:
    """
    Choose optimal metrics for a media item based on type and metadata.

    Args:
        media: Dict with keys id, media_type, media_product_type, timestamp, etc.

    Returns:
        List of metric names supported and high-impact for this media
    """
    media_id = media.get("id")
    media_type = media.get("media_type", "").upper()
    product_type = media.get("media_product_type", "").upper()

    # Try metadata first for precise metric discovery
    supported = fetch_media_insights_metadata(media_id)

    if supported:
        # Build prioritized metric list based on metadata
        chosen = []

        # Core engagement metrics (highest priority)
        for metric in ["total_interactions", "reach", "comments", "shares", "saved"]:
            if metric in supported:
                chosen.append(metric)

        # Video/Reels specific metrics
        if "REEL" in product_type:
            reel_metrics = [
                "ig_reels_avg_watch_time", "ig_reels_video_view_total_time", 
                "clips_replays_count", "ig_reels_aggregated_all_plays_count",
                "profile_visits", "follows", "navigation"
            ]
            for metric in reel_metrics:
                if metric in supported and metric not in chosen:
                    chosen.append(metric)
        elif media_type == "VIDEO":
            video_metrics = ["video_views", "profile_visits", "follows", "navigation"]
            for metric in video_metrics:
                if metric in supported and metric not in chosen:
                    chosen.append(metric)

        # Additional engagement metrics
        additional_metrics = ["website_clicks", "email_contacts", "phone_call_clicks"]
        for metric in additional_metrics:
            if metric in supported and metric not in chosen:
                chosen.append(metric)

        logger.info(f"🎯 Chosen metrics for media {media_id} ({media_type}/{product_type}): {chosen}")
        return chosen

    # Fallback based on media type when metadata unavailable
    media_key = product_type if product_type == "REEL" else media_type
    fallback = []
    
    for metric, supported_types in FALLBACK_IG_METRICS.items():
        if media_key in supported_types or media_type in supported_types:
            fallback.append(metric)

    # Prioritize core metrics
    priority_order = ["reach", "total_interactions", "comments", "shares", "saved", "profile_visits"]
    final_fallback = []
    
    for metric in priority_order:
        if metric in fallback:
            final_fallback.append(metric)
    
    for metric in fallback:
        if metric not in final_fallback:
            final_fallback.append(metric)

    logger.info(f"📋 No metadata for media {media_id}, using fallback metrics: {final_fallback}")
    return final_fallback

def fetch_insights_for_media(media: Dict) -> List[Dict]:
    """
    Optimized insights fetch for a single media item using safe API calls.

    Args:
        media: Dict with media information from /{ig_user_id}/media

    Returns:
        List of records with media info + metric + value
    """
    media_id = media.get("id")
    timestamp = media.get("timestamp")
    caption = media.get("caption", "")
    media_url = media.get("media_url")
    permalink = media.get("permalink")
    thumbnail_url = media.get("thumbnail_url")
    media_type = media.get("media_type")
    media_product_type = media.get("media_product_type")

    token, _ = get_page_access_token()
    records = []

    metrics = choose_metrics_for_media(media)
    if not metrics:
        logger.warning(f"No metrics available for media {media_id}")
        return records

    # Iterative removal approach for unsupported metrics with safe API calls
    to_try = metrics.copy()

    while to_try:
        metric_str = ",".join(to_try)
        
        def api_call():
            url = f"{GRAPH_API_BASE}/{media_id}/insights"
            params = {"metric": metric_str, "access_token": token}
            resp = requests.get(url, params=params, timeout=15)
            return resp

        # Use safe API call wrapper
        resp = safe_api_call(
            api_call,
            f"media_insights_{media_id}",
            {"metrics": metric_str},
            cache_ttl_hours=6,  # Cache media insights for 6 hours
            use_cache=True
        )

        if resp is None:
            logger.warning(f"❌ Rate limited or failed for media {media_id}")
            break

        try:
            body = resp.json() if resp.headers.get("Content-Type", "").startswith("application/json") else {}
        except:
            body = {}

        if resp.status_code == 200 and "data" in body:
            # Success - extract metrics
            for metric_obj in body["data"]:
                metric_name = metric_obj.get("name")
                values = metric_obj.get("values", [])

                if values:
                    # Use the most recent value
                    value = values[-1].get("value", 0)

                    records.append({
                        "media_id": media_id,
                        "timestamp": timestamp,
                        "caption": caption[:200] + "..." if len(caption) > 200 else caption,
                        "media_url": media_url,
                        "permalink": permalink,
                        "thumbnail_url": thumbnail_url,
                        "media_type": media_type,
                        "media_product_type": media_product_type,
                        "metric": metric_name,
                        "value": value
                    })

            logger.info(f"✅ Fetched {len(records)} metric records for media {media_id}")
            break

        elif resp.status_code == 400 and "error" in body:
            # Handle unsupported metric errors
            error_msg = body["error"].get("message", "")
            logger.warning(f"📱 Media {media_id} insights error: {error_msg}")

            # Try to identify problematic metric from error message
            removed_metric = None
            for metric in to_try:
                if metric.lower() in error_msg.lower():
                    removed_metric = metric
                    break

            if removed_metric:
                to_try.remove(removed_metric)
                logger.info(f"🔄 Removed unsupported metric '{removed_metric}', retrying with {to_try}")
                continue
            elif len(to_try) > 1:
                # Remove one metric and retry
                removed_metric = to_try.pop()
                logger.info(f"🔍 Couldn't identify problematic metric, removing '{removed_metric}' and retrying")
                continue
            else:
                logger.warning(f"❌ Last metric '{to_try[0]}' unsupported for media {media_id}")
                break
        else:
            logger.warning(f"❌ Insights fetch error for media {media_id}: status {resp.status_code}, {body}")
            break

    return records

def fetch_ig_media_insights(ig_user_id: str, since: Optional[str] = None, until: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch comprehensive Instagram media insights with metadata-driven discovery.

    Args:
        ig_user_id: Instagram Business User ID
        since: Start date in YYYY-MM-DD format (optional)
        until: End date in YYYY-MM-DD format (optional)

    Returns:
        DataFrame with comprehensive Instagram insights in long format
    """
    token, _ = get_page_access_token()

    if not ig_user_id or not token:
        logger.error("❌ Missing IG_USER_ID or token in fetch_ig_media_insights.")
        return pd.DataFrame(columns=[
            'media_id', 'timestamp', 'caption', 'media_url', 'permalink', 
            'thumbnail_url', 'media_type', 'media_product_type', 'metric', 'value'
        ])

    # Fetch media list with comprehensive fields
    url = f"{GRAPH_API_BASE}/{ig_user_id}/media"
    params = {
        "fields": "id,caption,timestamp,media_type,media_product_type,media_url,permalink,thumbnail_url",
        "access_token": token,
        "limit": 100
    }

    media_items = []

    try:
        # Handle pagination
        while url:
            logger.info(f"Fetching media list from: {url}")
            resp = requests.get(url, params=params, timeout=15)
            body = resp.json() if resp.headers.get("Content-Type", "").startswith("application/json") else {}
            
            logger.info(f"Media list API response: status={resp.status_code}")

            if resp.status_code != 200 or "error" in body:
                logger.error(f"❌ Error fetching IG media list: {body}")
                break

            data = body.get("data", [])
            media_items.extend(data)

            # Get next page
            paging = body.get("paging", {})
            url = paging.get("next")
            params = None  # Next URL includes all params

    except Exception as e:
        logger.error(f"❌ Exception fetching IG media list: {e}", exc_info=True)
        return pd.DataFrame(columns=[
            'media_id', 'timestamp', 'caption', 'media_url', 'permalink', 
            'thumbnail_url', 'media_type', 'media_product_type', 'metric', 'value'
        ])

    logger.info(f"📊 Found {len(media_items)} Instagram media items")

    # Filter by date range if specified
    filtered_media = []
    for media in media_items:
        timestamp = media.get("timestamp")
        if since or until:
            try:
                date_str = timestamp.split("T")[0] if timestamp else None
            except:
                date_str = None

            if since and date_str and date_str < since:
                continue
            if until and date_str and date_str > until:
                continue

        filtered_media.append(media)

    logger.info(f"🎯 {len(filtered_media)}/{len(media_items)} media items after date filter")

    # Fetch insights for each media item
    all_records = []
    for i, media in enumerate(filtered_media):
        logger.info(f"📱 Processing media {i+1}/{len(filtered_media)}: {media.get('id')}")
        records = fetch_insights_for_media(media)
        all_records.extend(records)

    if not all_records:
        logger.warning("⚠️ No IG media insights returned.")
        return pd.DataFrame(columns=[
            'media_id', 'timestamp', 'caption', 'media_url', 'permalink', 
            'thumbnail_url', 'media_type', 'media_product_type', 'metric', 'value'
        ])

    df = pd.DataFrame(all_records)
    logger.info(f"✅ Successfully fetched {len(df)} Instagram insights records")

    # Add computed date column
    df['date'] = pd.to_datetime(df['timestamp']).dt.date

    return df

def fetch_latest_ig_media_insights(ig_user_id: str) -> pd.DataFrame:
    """
    Fetch latest (yesterday's) Instagram media insights.

    Args:
        ig_user_id: Instagram Business User ID

    Returns:
        DataFrame with latest Instagram insights
    """
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"📅 Fetching latest Instagram insights for date: {yesterday}")
    return fetch_ig_media_insights(ig_user_id, since=yesterday, until=yesterday)

def compute_instagram_kpis(df: pd.DataFrame, follower_count: Optional[int] = None) -> Dict:
    """
    Compute advanced KPIs from Instagram insights data.

    Args:
        df: Instagram insights DataFrame (long format)
        follower_count: Account follower count for rate calculations

    Returns:
        Dict with computed KPIs and metrics
    """
    if df.empty:
        return {}

    kpis = {}

    # Aggregate metrics across all media
    metrics_summary = df.groupby('metric')['value'].sum().to_dict()

    # Core metrics
    total_reach = metrics_summary.get('reach', 0)
    total_interactions = metrics_summary.get('total_interactions', 0)
    total_comments = metrics_summary.get('comments', 0)
    total_shares = metrics_summary.get('shares', 0)
    total_saves = metrics_summary.get('saved', 0)

    # Engagement rates
    if follower_count and follower_count > 0:
        kpis['engagement_rate_by_followers'] = (total_interactions / follower_count) * 100

    if total_reach > 0:
        kpis['engagement_rate_by_reach'] = (total_interactions / total_reach) * 100
        kpis['save_rate'] = (total_saves / total_reach) * 100
        kpis['comment_rate'] = (total_comments / total_reach) * 100
        kpis['share_rate'] = (total_shares / total_reach) * 100

    # Video metrics
    total_video_views = metrics_summary.get('video_views', 0)
    total_reels_watch_time = metrics_summary.get('ig_reels_video_view_total_time', 0)
    avg_reels_watch_time = metrics_summary.get('ig_reels_avg_watch_time', 0)

    if total_video_views > 0 and total_reach > 0:
        kpis['video_view_rate'] = (total_video_views / total_reach) * 100

    # Growth metrics
    total_follows = metrics_summary.get('follows', 0)
    total_profile_visits = metrics_summary.get('profile_visits', 0)

    if total_reach > 0:
        kpis['profile_visit_rate'] = (total_profile_visits / total_reach) * 100
        kpis['follow_rate'] = (total_follows / total_reach) * 100

    # Summary totals
    kpis.update({
        'total_reach': total_reach,
        'total_interactions': total_interactions,
        'total_video_views': total_video_views,
        'total_profile_visits': total_profile_visits,
        'total_follows': total_follows,
        'total_saves': total_saves,
        'media_count': len(df['media_id'].unique()),
        'follower_count': follower_count,
        'avg_reels_watch_time': avg_reels_watch_time
    })

    return kpis

# Legacy compatibility functions
def get_cached_page_metrics():
    """Legacy function for page metrics - kept for compatibility."""
    return FALLBACK_PAGE_METRICS

def fetch_page_insights(metrics: List[str] = None, since: str = None, until: str = None, period: str = "day") -> pd.DataFrame:
    """Legacy page insights function - kept for compatibility."""
    return pd.DataFrame()

def get_organic_insights(date_preset: Optional[str] = None, since: Optional[str] = None, until: Optional[str] = None, 
                        metrics: Optional[List[str]] = None, include_instagram: bool = True) -> pd.DataFrame:
    """Legacy combined insights function - kept for compatibility."""
    validation = validate_organic_environment()

    if not include_instagram or not validation['instagram_insights_enabled']:
        return pd.DataFrame()

    # Handle date range
    if date_preset and not (since and until):
        if date_preset == "yesterday":
            yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
            since, until = yesterday, yesterday
        elif date_preset == "last_7d":
            end_date = date.today() - timedelta(days=1)
            start_date = end_date - timedelta(days=6)
            since = start_date.strftime("%Y-%m-%d")
            until = end_date.strftime("%Y-%m-%d")
        else:
            yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
            since, until = yesterday, yesterday

    ig_user_id = os.getenv('IG_USER_ID')
    if ig_user_id:
        return fetch_ig_media_insights(ig_user_id, since, until)

    return pd.DataFrame()

if __name__ == "__main__":
    # Test enhanced Instagram insights
    logger.info("🧪 Testing enhanced Instagram insights with metadata discovery...")

    ig_id = os.getenv("IG_USER_ID")
    if ig_id:
        # Test follower count
        follower_count = get_ig_follower_count(ig_id)
        print(f"📊 Follower count: {follower_count}")

        # Test latest insights
        df = fetch_latest_ig_media_insights(ig_id)
        print(f"📱 Latest insights: {len(df)} records")

        if not df.empty:
            print("📋 Columns:", df.columns.tolist())
            print("📈 Available metrics:", sorted(df['metric'].unique()))

            # Test KPI computation
            kpis = compute_instagram_kpis(df, follower_count)
            print("🎯 Computed KPIs:", kpis)

            # Show sample data
            print("\n📊 Sample data:")
            print(df.head(10))
    else:
        print("⚠️ IG_USER_ID not set - skipping tests")
