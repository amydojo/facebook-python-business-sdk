"""
Fetch organic content insights from Facebook Pages and Instagram.
Handles both Facebook Page insights and Instagram Business Account insights.

Official docs:
- Page Insights: https://developers.facebook.com/docs/graph-api/reference/page/insights/
- Instagram Insights: https://developers.facebook.com/docs/instagram-api/guides/insights/
- Token requirements: https://developers.facebook.com/docs/facebook-login/access-tokens

Updated Graph API version to v23.0 with dynamic metric discovery and robust fallback mechanisms
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

# Cache for page insights metadata
_cached_page_metrics = None
_metadata_fetch_failed = False

# Fallback Page metrics when metadata fetch fails
FALLBACK_PAGE_METRICS = ["page_impressions", "page_engaged_users", "page_reach"]

# Valid Instagram metrics for Graph API v23.0
# Official docs: https://developers.facebook.com/docs/instagram-api/reference/ig-media/insights
VALID_IG_METRICS = {
    "impressions", "reach", "replies", "saved", "video_views", "likes", "comments",
    "shares", "plays", "total_interactions", "follows", "profile_visits",
    "profile_activity", "navigation", "ig_reels_video_view_total_time",
    "ig_reels_avg_watch_time", "clips_replays_count", "ig_reels_aggregated_all_plays_count", "views"
}

# Default Instagram metrics (safe subset)
DEFAULT_IG_METRICS = ['impressions', 'reach', 'total_interactions']

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

def get_cached_page_metrics():
    """
    Get cached Page metrics metadata with robust fallback mechanism.

    Returns:
        List of available metric names or fallback metrics if metadata fetch fails
    """
    global _cached_page_metrics, _metadata_fetch_failed

    if _cached_page_metrics is not None:
        return _cached_page_metrics

    if _metadata_fetch_failed:
        logger.info("Using fallback Page metrics (metadata previously failed)")
        _cached_page_metrics = FALLBACK_PAGE_METRICS
        return _cached_page_metrics

    # Try metadata fetch
    page_id = os.getenv("PAGE_ID")
    token = os.getenv("PAGE_ACCESS_TOKEN") or os.getenv("META_ACCESS_TOKEN")

    if not page_id or not token:
        logger.error("Cannot fetch Page metadata: missing PAGE_ID or token")
        _metadata_fetch_failed = True
        _cached_page_metrics = FALLBACK_PAGE_METRICS
        return _cached_page_metrics

    url = f"{GRAPH_API_BASE}/{page_id}/insights/metadata"
    logger.info(f"Fetching Page insights metadata from: {url}")

    try:
        resp = requests.get(url, params={"access_token": token})
        try:
            body = resp.json()
        except ValueError:
            body = {"error": "Non-JSON response"}

        if resp.status_code != 200 or "error" in body:
            logger.error(f"Page insights metadata error: status {resp.status_code}, response JSON: {body}")
            _metadata_fetch_failed = True
            _cached_page_metrics = FALLBACK_PAGE_METRICS
            logger.warning(f"Using fallback Page metrics: {_cached_page_metrics}")
        else:
            data = body.get("data", [])
            metric_names = [item.get("name") for item in data if item.get("name")]
            logger.info(f"Fetched {len(metric_names)} Page metrics metadata: {metric_names[:10]} ...")
            _cached_page_metrics = metric_names

        return _cached_page_metrics

    except Exception as e:
        logger.error(f"Exception fetching Page metadata: {e}", exc_info=True)
        _metadata_fetch_failed = True
        _cached_page_metrics = FALLBACK_PAGE_METRICS
        logger.warning(f"Using fallback Page metrics: {_cached_page_metrics}")
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
        "page_impressions_organic",
        "page_impressions_paid", 
        "page_engaged_users",
        "page_reach",
        "page_post_engagements"
    ]

    selected = [m for m in candidates if m in available_metrics]
    skipped = [m for m in candidates if m not in available_metrics]

    # If no preferred metrics are available, use the first few available metrics
    if not selected and available_metrics:
        selected = available_metrics[:3]  # Take first 3 available
        logger.warning(f"No preferred metrics available, using: {selected}")

    logger.info(f"Default Page metrics selected: {selected}")
    if skipped:
        logger.debug(f"Skipped unavailable Page metrics: {skipped}")

    return selected

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

def fetch_page_insights(metrics: List[str] = None, since: str = None, until: str = None, period: str = "day") -> pd.DataFrame:
    """
    Fetch Facebook Page insights for specified date range with robust error handling.

    Args:
        metrics: List of metric names (optional, will use defaults if None)
        since: Start date in YYYY-MM-DD format
        until: End date in YYYY-MM-DD format  
        period: Time period ('day', 'week', 'days_28')

    Returns:
        DataFrame with insights data or empty DataFrame on error
    """
    page_id = os.getenv("PAGE_ID")
    token = os.getenv("PAGE_ACCESS_TOKEN") or os.getenv("META_ACCESS_TOKEN")

    if not page_id or not token:
        logger.error("fetch_page_insights: Missing PAGE_ID or token.")
        return pd.DataFrame()

    # Get available metrics with fallback
    available = get_cached_page_metrics()

    # If caller passed metrics, filter; if None, pick defaults
    if metrics:
        valid = [m for m in metrics if m in available]
    else:
        valid = select_default_page_metrics(available)

    if not valid:
        logger.error(f"No valid Page metrics to request: requested={metrics}, available={available}")
        return pd.DataFrame()

    metric_str = ",".join(valid)
    url = f"{GRAPH_API_BASE}/{page_id}/insights"
    params = {
        "metric": metric_str,
        "period": period,
        "since": since,
        "until": until,
        "access_token": token
    }

    logger.info(f"Fetching Page insights for {since} to {until} with metrics: {valid}")

    try:
        resp = requests.get(url, params=params)
        try:
            body = resp.json()
        except ValueError:
            body = {"error": "Non-JSON response"}

        if resp.status_code != 200 or "error" in body:
            logger.error(f"Page insights fetch error: status {resp.status_code}, response JSON: {body}")
            return pd.DataFrame()

        data = body.get("data", [])
        records = []

        for mobj in data:
            name = mobj.get("name")
            for v in mobj.get("values", []):
                records.append({
                    "date": v.get("end_time", since),
                    "metric": name,
                    "value": v.get("value", 0)
                })

        if not records:
            logger.info("fetch_page_insights: No records returned.")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date']).dt.date
        logger.info(f"Successfully fetched {len(df)} Page insights records")
        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"fetch_page_insights network error: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"fetch_page_insights unexpected error: {e}", exc_info=True)
        return pd.DataFrame()

def fetch_latest_page_insights(metrics: List[str] = None, period: str = "day") -> pd.DataFrame:
    """
    Fetch latest (yesterday's) Page insights.

    Args:
        metrics: List of metric names (optional)
        period: Time period ('day', 'week', 'days_28')

    Returns:
        DataFrame with latest insights data
    """
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"Fetching latest Page insights for date: {yesterday}")
    return fetch_page_insights(metrics=metrics, since=yesterday, until=yesterday, period=period)

def fetch_ig_media_insights(ig_user_id: str, since: Optional[str] = None, until: Optional[str] = None, metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fetch Instagram media insights in long-format DataFrame with robust error handling.

    Official docs: https://developers.facebook.com/docs/instagram-api/guides/insights/

    Args:
        ig_user_id: Instagram Business User ID
        since: Start date in YYYY-MM-DD format (optional)
        until: End date in YYYY-MM-DD format (optional)
        metrics: List of metrics (optional, defaults to safe subset)

    Returns:
        DataFrame with columns ['media_id', 'timestamp', 'caption', 'media_url', 'permalink', 'thumbnail_url', 'metric', 'value']
        Returns empty DataFrame with correct columns if fetch fails
    """
    token = os.getenv("PAGE_ACCESS_TOKEN") or os.getenv("META_ACCESS_TOKEN")

    if not ig_user_id or not token:
        logger.error("fetch_ig_media_insights: Missing IG_USER_ID or token.")
        return pd.DataFrame(columns=['media_id', 'timestamp', 'caption', 'media_url', 'permalink', 'thumbnail_url', 'metric', 'value'])

    # Determine initial metrics - default to safe subset
    default_metrics = ["impressions", "reach", "total_interactions"]
    req_metrics = metrics or default_metrics

    # Filter against VALID_IG_METRICS
    valid_initial = [m for m in req_metrics if m in VALID_IG_METRICS]
    if not valid_initial:
        logger.error(f"No valid IG metrics in requested {req_metrics}")
        return pd.DataFrame(columns=['media_id', 'timestamp', 'caption', 'media_url', 'permalink', 'thumbnail_url', 'metric', 'value'])

    logger.info(f"Initial valid Instagram metrics: {valid_initial}")

    # Fetch media list via Graph API
    url_media = f"{GRAPH_API_BASE}/{ig_user_id}/media"
    params_media = {
        "fields": "id,caption,timestamp,media_type,media_product_type,media_url,permalink,thumbnail_url",
        "access_token": token,
        "limit": 100
    }

    try:
        resp_media = requests.get(url_media, params=params_media)
        try:
            body_media = resp_media.json()
        except ValueError:
            body_media = {"error": "Non-JSON response"}

        if resp_media.status_code != 200 or "error" in body_media:
            logger.error(f"Error fetching IG media list: status {resp_media.status_code}, response: {body_media}")
            return pd.DataFrame(columns=['media_id', 'timestamp', 'caption', 'media_url', 'permalink', 'thumbnail_url', 'metric', 'value'])

        media_data = body_media.get("data", [])
        logger.info(f"Found {len(media_data)} Instagram media items")

    except Exception as e:
        logger.error(f"fetch_ig_media_insights: Exception fetching media list: {e}", exc_info=True)
        return pd.DataFrame(columns=['media_id', 'timestamp', 'caption', 'media_url', 'permalink', 'thumbnail_url', 'metric', 'value'])

    records = []

    for media in media_data:
        media_id = media.get("id")
        timestamp_str = media.get("timestamp", "")
        caption = media.get("caption", "")
        media_url = media.get("media_url", "")
        permalink = media.get("permalink", "")
        thumbnail_url = media.get("thumbnail_url", "")

        # Filter by date range if specified
        if since or until:
            try:
                media_date = timestamp_str.split("T")[0]  # Extract YYYY-MM-DD part
            except:
                media_date = None

            if since and media_date and media_date < since:
                continue
            if until and media_date and media_date > until:
                continue

        # Per-media retry logic: start with all valid metrics, remove unsupported ones
        metrics_for_media = list(valid_initial)

        while metrics_for_media:
            metric_str = ",".join(metrics_for_media)
            url_insights = f"{GRAPH_API_BASE}/{media_id}/insights"
            params_insights = {"metric": metric_str, "access_token": token}

            try:
                resp_insights = requests.get(url_insights, params=params_insights)
                try:
                    body_insights = resp_insights.json()
                except ValueError:
                    body_insights = {"error": "Non-JSON response"}

                if resp_insights.status_code == 200 and "data" in body_insights:
                    # Success - extract each metric's value and create long-format records
                    for metric_obj in body_insights.get("data", []):
                        metric_name = metric_obj.get("name")
                        values_list = metric_obj.get("values", [])

                        if values_list:
                            # Use the last (most recent) value
                            metric_value = values_list[-1].get("value", 0)

                            records.append({
                                "media_id": media_id,
                                "timestamp": timestamp_str,
                                "caption": caption[:100] + "..." if len(caption) > 100 else caption,
                                "media_url": media_url,
                                "permalink": permalink,
                                "thumbnail_url": thumbnail_url,
                                "metric": metric_name,
                                "value": metric_value
                            })

                    break  # Successfully processed this media

                # Handle 400 error indicating unsupported metric for this media type
                elif resp_insights.status_code == 400 and "error" in body_insights:
                    error_msg = body_insights["error"].get("message", "")

                    # Try to detect which metric caused the error by checking if metric name appears in error message
                    # e.g., "Media does not support the impressions metric for this media product type"
                    unsupported_metric = None
                    for metric in metrics_for_media:
                        if metric in error_msg:
                            unsupported_metric = metric
                            break

                    if unsupported_metric:
                        metrics_for_media.remove(unsupported_metric)
                        logger.info(f"Media {media_id}: removed unsupported metric '{unsupported_metric}' and retrying with {len(metrics_for_media)} remaining")

                        if not metrics_for_media:
                            logger.warning(f"Media {media_id}: no metrics left after removing unsupported ones, skipping")
                            break
                        continue
                    else:
                        # Couldn't identify the problematic metric, skip this media
                        logger.warning(f"Media {media_id}: Could not identify unsupported metric from error: {error_msg}")
                        break

                else:
                    # Other error, skip this media
                    logger.warning(f"Instagram insights fetch error for media {media_id}: status {resp_insights.status_code}, response: {body_insights}")
                    break

            except Exception as e:
                logger.error(f"Error fetching insights for media {media_id}: {e}")
                break

    if not records:
        logger.info("fetch_ig_media_insights: No media insights returned")
        return pd.DataFrame(columns=['media_id', 'timestamp', 'caption', 'media_url', 'permalink', 'thumbnail_url', 'metric', 'value'])

    df = pd.DataFrame(records)
    logger.info(f"Successfully fetched {len(df)} Instagram insights records in long format")
    return df

def fetch_latest_ig_media_insights(ig_user_id: str, metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fetch latest (yesterday's) Instagram media insights in long-format.

    Args:
        ig_user_id: Instagram Business User ID
        metrics: List of metrics (optional, defaults to safe subset)

    Returns:
        DataFrame with columns ['media_id', 'timestamp', 'caption', 'media_url', 'permalink', 'thumbnail_url', 'metric', 'value']
    """
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"Fetching latest Instagram insights for date: {yesterday}")
    return fetch_ig_media_insights(ig_user_id, since=yesterday, until=yesterday, metrics=metrics)

def get_organic_insights(date_preset: Optional[str] = None, since: Optional[str] = None, until: Optional[str] = None, metrics: Optional[List[str]] = None, include_instagram: bool = True) -> pd.DataFrame:
    """
    Get organic insights for Facebook Page and optionally Instagram with robust fallback mechanisms.

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
        # Get available Page metrics and select defaults (with fallback)
        available_page_metrics = get_cached_page_metrics()
        metrics = select_default_page_metrics(available_page_metrics)

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
    # Test metadata fetch with fallback
    logger.info("🧪 Testing Page metrics metadata fetch with fallback...")
    available = get_cached_page_metrics()
    print(f"Available Page metrics metadata: {available}")

    defaults = select_default_page_metrics(available)
    print(f"Default Page metrics: {defaults}")

    # Test latest Page insights
    if defaults:
        logger.info("🧪 Testing latest Page insights...")
        df_latest = fetch_latest_page_insights(defaults)
        print(f"Latest Page insights: {len(df_latest)} records")
        if not df_latest.empty:
            print(df_latest.head())

    # Test Instagram latest with comprehensive example
    ig_id = os.getenv("IG_USER_ID")
    if ig_id:
        logger.info("🧪 Testing latest Instagram insights (long-format)...")
        df_ig = fetch_latest_ig_media_insights(ig_id, metrics=["impressions", "reach", "total_interactions"])
        print(f"Latest IG insights: {len(df_ig)} records")
        print("Columns:", df_ig.columns.tolist())

        if not df_ig.empty:
            print("\nSample records:")
            print(df_ig.head())

            # Test filtering by metric (long-format usage)
            impressions_data = df_ig[df_ig['metric'] == 'impressions']
            print(f"\nImpressions records: {len(impressions_data)}")
            if not impressions_data.empty:
                print(impressions_data[['media_id', 'value']])

            reach_data = df_ig[df_ig['metric'] == 'reach']
            print(f"\nReach records: {len(reach_data)}")

            # Show metrics per media example
            media_ids = df_ig['media_id'].unique()
            if len(media_ids) > 0:
                first_media = media_ids[0]
                media_metrics = df_ig[df_ig['media_id'] == first_media]
                print(f"\nMetrics for media {first_media}:")
                print(media_metrics[['metric', 'value', 'timestamp']])

            # Example pivot for dashboard use
            try:
                pivot_example = df_ig.pivot_table(
                    index='media_id', 
                    columns='metric', 
                    values='value', 
                    aggfunc='first'
                ).fillna(0)
                print(f"\nPivot table example (media_id x metrics):")
                print(pivot_example.head())
            except Exception as e:
                print(f"Pivot example failed: {e}")
        else:
            print("No Instagram insights data returned")

    # Test combined organic insights
    logger.info("🧪 Testing combined organic insights...")
    combined_df = get_organic_insights(date_preset="yesterday")
    print(f"Combined organic insights: {len(combined_df)} records")
    if not combined_df.empty:
        print(combined_df.head())

    # Interactive REPL test snippet (uncomment to use)
    """
    # For interactive testing, set your tokens:
    # os.environ["PAGE_ACCESS_TOKEN"] = "<your_page_token>"
    # os.environ["IG_USER_ID"] = "<your_ig_user_id>"
    # df_test = fetch_latest_ig_media_insights(os.getenv("IG_USER_ID"), metrics=["impressions","reach"])
    # print(df_test.head())
    # print("Columns:", df_test.columns.tolist())
    """