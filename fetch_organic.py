
"""
Fetch organic content insights from Facebook Pages and Instagram.
Handles both Facebook Page insights and Instagram Business Account insights.

Official docs:
- Page Insights: https://developers.facebook.com/docs/graph-api/reference/page/insights/
- Instagram Insights: https://developers.facebook.com/docs/instagram-api/guides/insights/
- Token requirements: https://developers.facebook.com/docs/facebook-login/access-tokens

Updated Graph API version to v23.0 with dynamic metric discovery
"""
import os
import requests
import pandas as pd
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Union
from config import config

logger = logging.getLogger(__name__)

# Graph API version and base URL for consistent endpoint calls
GRAPH_API_VERSION = "v23.0"
GRAPH_API_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"

# Valid Instagram metrics for Graph API v23.0
# Official docs: https://developers.facebook.com/docs/instagram-api/reference/ig-media/insights
VALID_IG_METRICS = {
    'impressions', 'reach', 'replies', 'saved', 'video_views', 'likes', 
    'comments', 'shares', 'plays', 'total_interactions', 'follows',
    'profile_visits', 'profile_activity', 'navigation', 
    'ig_reels_video_view_total_time', 'ig_reels_avg_watch_time',
    'clips_replays_count', 'ig_reels_aggregated_all_plays_count', 'views'
}

# Default Instagram metrics (safe subset)
DEFAULT_IG_METRICS = ['impressions', 'reach', 'total_interactions']

# Cache for page insights metadata
_cached_page_metrics = None

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

def fetch_page_insights_metadata():
    """
    Fetch available metrics for Page insights dynamically.
    
    Official docs: https://developers.facebook.com/docs/graph-api/reference/page/insights/metadata/
    Use this endpoint to list valid Page Insights metrics under the given API version.
    
    Returns:
        List of available metric names or empty list on error
    """
    page_token, _ = get_page_access_token()
    page_id = os.getenv('PAGE_ID')

    if not page_token or not page_id:
        logger.error("Cannot fetch Page metadata: missing PAGE_ACCESS_TOKEN or PAGE_ID")
        return []

    url = f"{GRAPH_API_BASE}/{page_id}/insights/metadata"
    params = {
        'access_token': page_token
    }

    try:
        logger.info(f"Fetching Page insights metadata from: {url}")
        resp = requests.get(url, params=params)
        body = resp.json()

        if resp.status_code != 200 or "error" in body:
            logger.error(f"Page insights metadata error: status {resp.status_code}, response JSON: {body}")
            return []

        # Extract metric names from metadata
        metric_names = []
        for item in body.get('data', []):
            if 'name' in item:
                metric_names.append(item['name'])

        logger.info(f"Fetched {len(metric_names)} Page metrics metadata: {metric_names[:10]} ...")
        return metric_names

    except Exception as e:
        logger.error(f"Error fetching Page insights metadata: {e}")
        return []

def get_cached_page_metrics():
    """
    Get cached Page metrics metadata, fetching if not cached.
    
    Returns:
        List of available metric names
    """
    global _cached_page_metrics
    
    if _cached_page_metrics is None:
        logger.info("Caching Page metrics metadata")
        _cached_page_metrics = fetch_page_insights_metadata()
    
    return _cached_page_metrics

def select_default_page_metrics(available_metrics: List[str]) -> List[str]:
    """
    Select default Page metrics from available metrics.
    
    Args:
        available_metrics: List of available metric names
    
    Returns:
        List of default metrics that are available
    """
    # Preferred default metrics (in order of preference)
    candidates = [
        'page_impressions_organic',
        'page_impressions_paid', 
        'page_engaged_users',
        'page_reach',
        'page_post_engagements'
    ]
    
    selected_metrics = []
    skipped_metrics = []
    
    for metric in candidates:
        if metric in available_metrics:
            selected_metrics.append(metric)
        else:
            skipped_metrics.append(metric)
    
    # If no preferred metrics are available, use the first few available metrics
    if not selected_metrics and available_metrics:
        selected_metrics = available_metrics[:3]  # Take first 3 available
        logger.warning(f"No preferred metrics available, using: {selected_metrics}")
    
    logger.info(f"Default Page metrics: {selected_metrics}")
    logger.debug(f"Skipped Page metrics (not available): {skipped_metrics}")
    
    return selected_metrics

def filter_valid_instagram_metrics(requested_metrics: List[str]) -> List[str]:
    """
    Filter Instagram metrics to only include valid ones.
    
    Args:
        requested_metrics: List of requested metric names
    
    Returns:
        List of valid metrics only
    """
    valid_metrics = [m for m in requested_metrics if m in VALID_IG_METRICS]
    invalid_metrics = [m for m in requested_metrics if m not in VALID_IG_METRICS]
    
    if invalid_metrics:
        logger.warning(f"Filtered out invalid Instagram metrics: {invalid_metrics}")
        logger.info(f"Valid Instagram metrics: {list(VALID_IG_METRICS)}")
    
    if not valid_metrics:
        logger.warning(f"No valid Instagram metrics from request, using defaults: {DEFAULT_IG_METRICS}")
        return DEFAULT_IG_METRICS
    
    logger.info(f"Using valid Instagram metrics: {valid_metrics}")
    return valid_metrics

def calculate_date_range(date_preset: str) -> tuple:
    """
    Calculate date range for given preset.
    Compute yesterday via datetime.date.today() - timedelta(days=1)
    
    Args:
        date_preset: Preset name
    
    Returns:
        Tuple of (since_date, until_date) as strings
    """
    today = date.today()
    
    if date_preset in ['latest', 'yesterday']:
        target_date = today - timedelta(days=1)
        return target_date.strftime('%Y-%m-%d'), target_date.strftime('%Y-%m-%d')
    
    elif date_preset == 'last_7d':
        end_date = today - timedelta(days=1)
        start_date = end_date - timedelta(days=6)
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    elif date_preset == 'last_30d':
        end_date = today - timedelta(days=1)
        start_date = end_date - timedelta(days=29)
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    elif date_preset == 'this_month':
        start_date = today.replace(day=1)
        return start_date.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')
    
    elif date_preset == 'last_month':
        # Last day of previous month
        first_this_month = today.replace(day=1)
        last_last_month = first_this_month - timedelta(days=1)
        first_last_month = last_last_month.replace(day=1)
        return first_last_month.strftime('%Y-%m-%d'), last_last_month.strftime('%Y-%m-%d')
    
    else:
        logger.error(f"Unknown date preset: {date_preset}")
        # Default to yesterday
        target_date = today - timedelta(days=1)
        return target_date.strftime('%Y-%m-%d'), target_date.strftime('%Y-%m-%d')

def fetch_page_insights(metrics: List[str], since: str, until: str, period: str = "day") -> pd.DataFrame:
    """
    Fetch Facebook Page insights for specified date range with dynamic metric validation.
    
    Args:
        metrics: List of metric names
        since: Start date in YYYY-MM-DD format
        until: End date in YYYY-MM-DD format  
        period: Time period ('day', 'week', 'days_28')

    Returns:
        DataFrame with insights data or empty DataFrame on error
    """
    page_token, _ = get_page_access_token()
    page_id = os.getenv('PAGE_ID')

    if not page_token or not page_id:
        logger.error("Cannot fetch page insights: missing PAGE_ACCESS_TOKEN or PAGE_ID")
        return pd.DataFrame()

    # Get available metrics and filter requested metrics
    available_metrics = get_cached_page_metrics()
    if not available_metrics:
        logger.error("Could not determine available Page metrics, skipping Page insights")
        return pd.DataFrame()
    
    # Filter metrics to only include available ones
    valid_metrics = [m for m in metrics if m in available_metrics]
    
    if not valid_metrics:
        logger.error(f"No valid Page metrics to request: {metrics}, available: {available_metrics}")
        return pd.DataFrame()

    metric_str = ",".join(valid_metrics)
    url = f"{GRAPH_API_BASE}/{page_id}/insights"
    params = {
        'metric': metric_str,
        'period': period,
        'since': since,
        'until': until,
        'access_token': page_token
    }

    try:
        logger.info(f"Fetching Page insights for {since} to {until} with metrics: {valid_metrics}")
        resp = requests.get(url, params=params)
        body = resp.json()

        if resp.status_code != 200 or "error" in body:
            logger.error(f"Page insights fetch error: status {resp.status_code}, response JSON: {body}")
            return pd.DataFrame()

        # Parse insights data
        insights_data = []
        for metric_data in body.get('data', []):
            metric_name = metric_data.get('name')
            for value_entry in metric_data.get('values', []):
                insights_data.append({
                    'date': value_entry.get('end_time', since),
                    'metric': metric_name,
                    'value': value_entry.get('value', 0)
                })

        df = pd.DataFrame(insights_data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date']).dt.date
            logger.info(f"Successfully fetched {len(df)} Page insights records")
        else:
            logger.warning("No Page insights data returned")

        return df

    except Exception as e:
        logger.error(f"Error fetching Page insights: {e}")
        return pd.DataFrame()

def fetch_latest_page_insights(metrics: List[str], period: str = "day") -> pd.DataFrame:
    """
    Fetch latest (yesterday's) Page insights.
    
    Args:
        metrics: List of metric names
        period: Time period ('day', 'week', 'days_28')
    
    Returns:
        DataFrame with latest insights data
    """
    # Compute yesterday via datetime.date.today() - timedelta(days=1)
    yesterday = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    logger.info(f"Fetching latest Page insights for date: {yesterday}")
    
    return fetch_page_insights(metrics, since=yesterday, until=yesterday, period=period)

def fetch_ig_media_insights(ig_user_id: str, since: Optional[str] = None, until: Optional[str] = None, metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fetch Instagram media insights for specified date range with metric validation.

    Args:
        ig_user_id: Instagram Business User ID
        since: Start date in YYYY-MM-DD format (optional)
        until: End date in YYYY-MM-DD format (optional)
        metrics: List of metrics (default: filtered defaults)

    Returns:
        DataFrame with Instagram insights data
    """
    page_token, _ = get_page_access_token()

    if not page_token:
        logger.error("Cannot fetch Instagram insights: missing PAGE_ACCESS_TOKEN")
        return pd.DataFrame()

    # Filter and validate metrics
    if not metrics:
        metrics = DEFAULT_IG_METRICS.copy()
    
    valid_metrics = filter_valid_instagram_metrics(metrics)
    if not valid_metrics:
        logger.error("No valid Instagram metrics available")
        return pd.DataFrame()

    # First, get media list
    media_url = f"{GRAPH_API_BASE}/{ig_user_id}/media"
    media_params = {
        'access_token': page_token,
        'fields': 'id,timestamp,media_type,caption'
    }

    try:
        logger.info(f"Fetching Instagram media from: {media_url}")
        media_resp = requests.get(media_url, params=media_params)
        media_body = media_resp.json()

        if media_resp.status_code != 200 or "error" in media_body:
            logger.error(f"Instagram media fetch error: status {media_resp.status_code}, response JSON: {media_body}")
            return pd.DataFrame()

        media_items = media_body.get('data', [])
        logger.info(f"Found {len(media_items)} Instagram media items")

        # Filter by date range if specified
        if since or until:
            filtered_media = []
            for item in media_items:
                item_date = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')).date()

                if since and item_date < datetime.strptime(since, '%Y-%m-%d').date():
                    continue
                if until and item_date > datetime.strptime(until, '%Y-%m-%d').date():
                    continue

                filtered_media.append(item)

            media_items = filtered_media
            logger.info(f"After date filtering: {len(media_items)} media items")

        # Fetch insights for each media item
        insights_data = []
        for media_item in media_items:
            media_id = media_item['id']
            insights_url = f"{GRAPH_API_BASE}/{media_id}/insights"
            insights_params = {
                'access_token': page_token,
                'metric': ','.join(valid_metrics)
            }

            try:
                insights_resp = requests.get(insights_url, params=insights_params)
                insights_body = insights_resp.json()

                if insights_resp.status_code != 200 or "error" in insights_body:
                    logger.error(f"Instagram insights fetch error for media {media_id}: status {insights_resp.status_code}, response JSON: {insights_body}")
                    continue

                # Parse insights data
                for metric_data in insights_body.get('data', []):
                    insights_data.append({
                        'media_id': media_id,
                        'date': datetime.fromisoformat(media_item['timestamp'].replace('Z', '+00:00')).date(),
                        'metric': metric_data.get('name'),
                        'value': metric_data.get('values', [{}])[0].get('value', 0),
                        'media_type': media_item.get('media_type'),
                        'caption': media_item.get('caption', '')[:100] + '...' if len(media_item.get('caption', '')) > 100 else media_item.get('caption', '')
                    })

            except Exception as e:
                logger.error(f"Error fetching insights for media {media_id}: {e}")
                continue

        df = pd.DataFrame(insights_data)
        if not df.empty:
            logger.info(f"Successfully fetched {len(df)} Instagram insights records")
        else:
            logger.warning("No Instagram insights data returned")

        return df

    except Exception as e:
        logger.error(f"Error fetching Instagram media insights: {e}")
        return pd.DataFrame()

def fetch_latest_ig_media_insights(ig_user_id: str, metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fetch latest (yesterday's) Instagram media insights.
    
    Official docs: https://developers.facebook.com/docs/instagram-api/guides/insights/
    Fetch yesterday's Instagram media insights by filtering media timestamps.
    
    Args:
        ig_user_id: Instagram Business User ID
        metrics: List of metrics (optional, defaults to safe subset)
    
    Returns:
        DataFrame with latest Instagram insights data
    """
    # Compute yesterday via datetime.date.today() - timedelta(days=1)
    yesterday = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    logger.info(f"Fetching latest Instagram insights for date: {yesterday}")
    
    return fetch_ig_media_insights(ig_user_id, since=yesterday, until=yesterday, metrics=metrics)

def get_organic_insights(date_preset: Optional[str] = None, since: Optional[str] = None, until: Optional[str] = None, metrics: Optional[List[str]] = None, include_instagram: bool = True) -> pd.DataFrame:
    """
    Get organic insights for Facebook Page and optionally Instagram with dynamic metric discovery.

    Args:
        date_preset: Preset date range ('latest', 'yesterday', 'last_7d', 'last_30d', 'this_month', 'last_month')
        since: Custom start date (YYYY-MM-DD)
        until: Custom end date (YYYY-MM-DD)
        metrics: List of metrics to fetch (will be filtered for validity)
        include_instagram: Whether to include Instagram insights

    Returns:
        Combined DataFrame with organic insights data
    """
    validation = validate_organic_environment()

    # Handle date range calculation
    if date_preset and not (since and until):
        since, until = calculate_date_range(date_preset)
        logger.info(f"Using date preset '{date_preset}': {since} to {until}")
    elif not (since and until):
        logger.error("No valid date range specified")
        return pd.DataFrame()

    # Get default metrics if none specified, then validate them
    if not metrics:
        # Get available Page metrics and select defaults
        available_page_metrics = get_cached_page_metrics()
        if available_page_metrics:
            metrics = select_default_page_metrics(available_page_metrics)
        else:
            logger.warning("Could not get available Page metrics, using fallback")
            metrics = ['page_views']  # Fallback metric

    # Fetch Page insights
    page_df = pd.DataFrame()
    if validation['page_insights_enabled']:
        logger.info(f"Fetching Page insights for {since} to {until}")
        page_df = fetch_page_insights(metrics, since, until)
    else:
        logger.warning("Page insights disabled - missing PAGE_ACCESS_TOKEN or PAGE_ID")

    # Fetch Instagram insights
    ig_df = pd.DataFrame()
    if include_instagram and validation['instagram_insights_enabled']:
        logger.info(f"Fetching Instagram insights for {since} to {until}")
        ig_user_id = os.getenv('IG_USER_ID')
        ig_df = fetch_ig_media_insights(ig_user_id, since, until, DEFAULT_IG_METRICS)
    elif include_instagram and validation['ig_user_id_available']:
        logger.warning("Instagram insights disabled - missing PAGE_ACCESS_TOKEN")

    # Combine Page and Instagram data
    combined_data = []

    if not page_df.empty:
        page_df['source'] = 'facebook_page'
        combined_data.append(page_df)

    if not ig_df.empty:
        ig_df['source'] = 'instagram'
        combined_data.append(ig_df)

    if combined_data:
        result_df = pd.concat(combined_data, ignore_index=True)
        logger.info(f"Combined organic insights: {len(result_df)} total records")
        return result_df
    else:
        logger.warning("No organic insights data available")
        return pd.DataFrame()

def get_organic_performance_summary(date_preset: str = "last_7d") -> Dict:
    """
    Get summary metrics for organic performance.

    Args:
        date_preset: Date range preset

    Returns:
        Dict with summary metrics
    """
    df = get_organic_insights(date_preset=date_preset)

    if df.empty:
        return {
            "total_reach": 0,
            "total_impressions": 0,
            "total_engagement": 0,
            "avg_engagement_rate": 0.0
        }

    # Calculate summary metrics from available data
    page_data = df[df['source'] == 'facebook_page'] if 'source' in df.columns else df

    # Try different metric names based on what's available
    total_reach = 0
    total_impressions = 0
    total_engagement = 0
    
    if not page_data.empty:
        # Try different reach metrics
        reach_metrics = ['page_reach', 'page_impressions', 'page_views']
        for metric in reach_metrics:
            metric_data = page_data[page_data['metric'] == metric]['value'].sum()
            if metric_data > 0:
                total_reach = metric_data
                break
        
        # Try different impression metrics
        impression_metrics = ['page_impressions_organic', 'page_impressions', 'page_views']
        for metric in impression_metrics:
            metric_data = page_data[page_data['metric'] == metric]['value'].sum()
            if metric_data > 0:
                total_impressions = metric_data
                break
        
        # Try engagement metrics
        engagement_metrics = ['page_post_engagements', 'page_engaged_users']
        for metric in engagement_metrics:
            metric_data = page_data[page_data['metric'] == metric]['value'].sum()
            if metric_data > 0:
                total_engagement = metric_data
                break

    # Calculate engagement rate
    avg_engagement_rate = (total_engagement / total_impressions * 100) if total_impressions > 0 else 0.0

    summary = {
        "total_reach": int(total_reach),
        "total_impressions": int(total_impressions),
        "total_engagement": int(total_engagement),
        "avg_engagement_rate": round(avg_engagement_rate, 2)
    }

    logger.info(f"Organic performance summary: {summary}")
    return summary

# Enhanced preset mapping for date ranges
ORGANIC_DATE_PRESETS = {
    "latest": "Latest (Yesterday)",
    "yesterday": "Yesterday", 
    "last_7d": "Last 7 Days",
    "last_30d": "Last 30 Days",
    "this_month": "This Month",
    "last_month": "Last Month",
    "custom": "Custom Range"
}

# Export helper functions for dashboard use
def get_available_page_metrics():
    """Get list of available Page metrics for dashboard display."""
    return get_cached_page_metrics()

def get_valid_instagram_metrics():
    """Get list of valid Instagram metrics for dashboard display."""
    return list(VALID_IG_METRICS)

if __name__ == "__main__":
    # Test metadata fetch
    available = get_cached_page_metrics()
    print("Available Page metrics metadata:", available)
    defaults = select_default_page_metrics(available)
    print("Default Page metrics:", defaults)
    # Test latest Page insights
    if defaults:
        df_latest = fetch_latest_page_insights(defaults)
        print("Latest Page insights:", df_latest)
    # Test Instagram latest
    ig_id = os.getenv("IG_USER_ID")
    if ig_id:
        df_ig = fetch_latest_ig_media_insights(ig_id, metrics=["impressions","reach"])
        print("Latest IG insights:", df_ig)
