
"""
Fetch organic content insights from Facebook Pages and Instagram.
Handles both Facebook Page insights and Instagram Business Account insights.

Official docs:
- Page Insights: https://developers.facebook.com/docs/graph-api/reference/page/insights/
- Instagram Insights: https://developers.facebook.com/docs/instagram-api/guides/insights/
- Token requirements: https://developers.facebook.com/docs/facebook-login/access-tokens
"""
import os
import requests
import pandas as pd
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Union
from config import config

logger = logging.getLogger(__name__)

# Graph API version for consistent endpoint calls
GRAPH_API_VERSION = os.getenv('GRAPH_API_VERSION', 'v18.0')

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
    Fetch available metrics for Page insights to help with debugging.
    Official docs: https://developers.facebook.com/docs/graph-api/reference/page/insights/
    """
    page_token, _ = get_page_access_token()
    page_id = os.getenv('PAGE_ID')
    
    if not page_token or not page_id:
        logger.error("Cannot fetch metadata: missing PAGE_ACCESS_TOKEN or PAGE_ID")
        return None
    
    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{page_id}/insights/metadata"
    params = {'access_token': page_token}
    
    try:
        logger.info(f"Fetching Page insights metadata from: {url}")
        resp = requests.get(url, params=params)
        body = resp.json()
        
        if resp.status_code != 200:
            logger.error(f"Page insights metadata error: status {resp.status_code}, response JSON: {body}")
            return None
        
        logger.info(f"Available Page insights metrics: {[metric.get('name') for metric in body.get('data', [])]}")
        return body
        
    except Exception as e:
        logger.error(f"Error fetching Page insights metadata: {e}")
        return None

def fetch_page_insights(metrics: List[str], since: str, until: str, period: str = "day") -> pd.DataFrame:
    """
    Fetch Facebook Page insights for specified date range.
    
    Args:
        metrics: List of metric names (e.g., ['page_impressions_organic', 'page_engaged_users'])
        since: Start date in YYYY-MM-DD format
        until: End date in YYYY-MM-DD format  
        period: Time period ('day', 'week', 'days_28')
    
    Returns:
        DataFrame with insights data or empty DataFrame on error
        
    Official docs: https://developers.facebook.com/docs/graph-api/reference/page/insights/
    """
    page_token, _ = get_page_access_token()
    page_id = os.getenv('PAGE_ID')
    
    if not page_token or not page_id:
        logger.error("Cannot fetch page insights: missing PAGE_ACCESS_TOKEN or PAGE_ID")
        return pd.DataFrame()
    
    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{page_id}/insights"
    params = {
        'access_token': page_token,
        'metric': ','.join(metrics),
        'since': since,
        'until': until,
        'period': period
    }
    
    try:
        logger.info(f"Fetching Page insights from: {url} with params: {params}")
        resp = requests.get(url, params=params)
        body = resp.json()
        
        if resp.status_code != 200:
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
    Fetch Facebook Page insights for yesterday (most recent complete day).
    
    Args:
        metrics: List of metric names
        period: Time period (default 'day')
    
    Returns:
        DataFrame with latest insights data
        
    # Compute yesterday via datetime.date.today() - timedelta(days=1)
    """
    # Calculate yesterday's date
    yesterday = date.today() - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    
    logger.info(f"Fetching latest Page insights for date: {yesterday_str}")
    
    # For single day, since and until are the same
    return fetch_page_insights(metrics, yesterday_str, yesterday_str, period)

def fetch_ig_media_insights(ig_user_id: str, since: Optional[str] = None, until: Optional[str] = None, metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fetch Instagram media insights for specified date range.
    
    Args:
        ig_user_id: Instagram Business User ID
        since: Start date in YYYY-MM-DD format (optional)
        until: End date in YYYY-MM-DD format (optional)
        metrics: List of metrics (default: ['impressions', 'reach', 'engagement'])
    
    Returns:
        DataFrame with Instagram insights data
        
    Official docs: https://developers.facebook.com/docs/instagram-api/guides/insights/
    """
    page_token, _ = get_page_access_token()
    
    if not page_token:
        logger.error("Cannot fetch Instagram insights: missing PAGE_ACCESS_TOKEN")
        return pd.DataFrame()
    
    if not metrics:
        metrics = ['impressions', 'reach', 'engagement']
    
    # First, get media list
    media_url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{ig_user_id}/media"
    media_params = {
        'access_token': page_token,
        'fields': 'id,timestamp,media_type,caption'
    }
    
    try:
        logger.info(f"Fetching Instagram media from: {media_url}")
        media_resp = requests.get(media_url, params=media_params)
        media_body = media_resp.json()
        
        if media_resp.status_code != 200:
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
            insights_url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{media_id}/insights"
            insights_params = {
                'access_token': page_token,
                'metric': ','.join(metrics)
            }
            
            try:
                insights_resp = requests.get(insights_url, params=insights_params)
                insights_body = insights_resp.json()
                
                if insights_resp.status_code != 200:
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
    Fetch Instagram media insights for yesterday (most recent complete day).
    
    Args:
        ig_user_id: Instagram Business User ID
        metrics: List of metrics (optional)
    
    Returns:
        DataFrame with latest Instagram insights data
        
    # Compute yesterday via datetime.date.today() - timedelta(days=1)
    """
    # Calculate yesterday's date
    yesterday = date.today() - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    
    logger.info(f"Fetching latest Instagram insights for date: {yesterday_str}")
    
    return fetch_ig_media_insights(ig_user_id, yesterday_str, yesterday_str, metrics)

def get_organic_insights(date_preset: Optional[str] = None, since: Optional[str] = None, until: Optional[str] = None, metrics: Optional[List[str]] = None, include_instagram: bool = True) -> pd.DataFrame:
    """
    Get organic insights for Facebook Page and optionally Instagram.
    
    Args:
        date_preset: Preset date range ('latest', 'yesterday', 'last_7d', 'last_30d', etc.)
        since: Custom start date (YYYY-MM-DD)
        until: Custom end date (YYYY-MM-DD)
        metrics: List of metrics to fetch
        include_instagram: Whether to include Instagram insights
    
    Returns:
        Combined DataFrame with organic insights data
        
    Official docs: https://developers.facebook.com/docs/graph-api/reference/page/insights/
    """
    validation = validate_organic_environment()
    
    if not metrics:
        metrics = [
            'page_impressions_organic',
            'page_engaged_users',
            'page_reach',
            'page_post_engagements'
        ]
    
    # Handle date presets
    if date_preset in ['latest', 'yesterday']:
        logger.info(f"Using date preset: {date_preset}")
        if validation['page_insights_enabled']:
            page_df = fetch_latest_page_insights(metrics)
        else:
            logger.warning("Page insights disabled - missing PAGE_ACCESS_TOKEN or PAGE_ID")
            page_df = pd.DataFrame()
        
        # Instagram latest insights
        ig_df = pd.DataFrame()
        if include_instagram and validation['instagram_insights_enabled']:
            ig_user_id = os.getenv('IG_USER_ID')
            ig_df = fetch_latest_ig_media_insights(ig_user_id, ['impressions', 'reach', 'engagement'])
        elif include_instagram and validation['ig_user_id_available']:
            logger.warning("Instagram insights disabled - missing PAGE_ACCESS_TOKEN")
        
    elif date_preset == 'last_7d':
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=6)
        since = start_date.strftime('%Y-%m-%d')
        until = end_date.strftime('%Y-%m-%d')
        
        if validation['page_insights_enabled']:
            page_df = fetch_page_insights(metrics, since, until)
        else:
            page_df = pd.DataFrame()
        
        ig_df = pd.DataFrame()
        if include_instagram and validation['instagram_insights_enabled']:
            ig_user_id = os.getenv('IG_USER_ID')
            ig_df = fetch_ig_media_insights(ig_user_id, since, until)
        
    elif date_preset == 'last_30d':
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=29)
        since = start_date.strftime('%Y-%m-%d')
        until = end_date.strftime('%Y-%m-%d')
        
        if validation['page_insights_enabled']:
            page_df = fetch_page_insights(metrics, since, until)
        else:
            page_df = pd.DataFrame()
        
        ig_df = pd.DataFrame()
        if include_instagram and validation['instagram_insights_enabled']:
            ig_user_id = os.getenv('IG_USER_ID')
            ig_df = fetch_ig_media_insights(ig_user_id, since, until)
    
    elif since and until:
        # Custom date range
        if validation['page_insights_enabled']:
            page_df = fetch_page_insights(metrics, since, until)
        else:
            page_df = pd.DataFrame()
        
        ig_df = pd.DataFrame()
        if include_instagram and validation['instagram_insights_enabled']:
            ig_user_id = os.getenv('IG_USER_ID')
            ig_df = fetch_ig_media_insights(ig_user_id, since, until)
    
    else:
        logger.error("No valid date range specified")
        return pd.DataFrame()
    
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
    
    # Calculate summary metrics
    page_data = df[df['source'] == 'facebook_page'] if 'source' in df.columns else df
    
    total_reach = page_data[page_data['metric'] == 'page_reach']['value'].sum() if not page_data.empty else 0
    total_impressions = page_data[page_data['metric'] == 'page_impressions_organic']['value'].sum() if not page_data.empty else 0
    total_engagement = page_data[page_data['metric'] == 'page_post_engagements']['value'].sum() if not page_data.empty else 0
    
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

# Preset mapping for date ranges
ORGANIC_DATE_PRESETS = {
    "latest": "Latest (Yesterday)",
    "yesterday": "Yesterday", 
    "last_7d": "Last 7 Days",
    "last_30d": "Last 30 Days",
    "this_month": "This Month",
    "last_month": "Last Month"
}
