"""
Fetch organic insights from Facebook Pages and Instagram.
Official docs: https://developers.facebook.com/docs/graph-api/reference/page/insights/
"""
import logging
import pandas as pd
import requests
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

def fetch_page_insights(metrics=None, since=None, until=None, period='day'):
    """
    Fetch Facebook Page insights using Graph API.

    Args:
        metrics: list of metrics to fetch
        since: start date (YYYY-MM-DD)
        until: end date (YYYY-MM-DD)
        period: 'day', 'week', 'days_28'

    Returns:
        pandas.DataFrame with page insights

    Official docs: https://developers.facebook.com/docs/graph-api/reference/page/insights/
    """
    page_id = os.getenv("PAGE_ID")
    access_token = os.getenv("META_ACCESS_TOKEN")

    if not page_id or not access_token:
        logger.error("❌ PAGE_ID or META_ACCESS_TOKEN not configured")
        return pd.DataFrame()

    # Default metrics if none provided
    if metrics is None:
        metrics = [
            'page_impressions', 'page_reach', 'page_engaged_users',
            'page_post_engagements', 'page_fans', 'page_fan_adds',
            'page_fan_removes', 'page_views_total', 'page_actions_post_reactions_total'
        ]

    # Default date range to last 7 days if not provided
    if not since or not until:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)
        since = start_date.strftime('%Y-%m-%d')
        until = end_date.strftime('%Y-%m-%d')

    try:
        all_data = []
        api_version = "v18.0"

        # Fetch each metric separately (API limitation)
        for metric in metrics:
            logger.info(f"🔍 Fetching page metric: {metric}")

            url = f"https://graph.facebook.com/{api_version}/{page_id}/insights"
            params = {
                'metric': metric,
                'period': period,
                'since': since,
                'until': until,
                'access_token': access_token
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if 'data' in data and data['data']:
                metric_data = data['data'][0]

                # Extract values by date
                if 'values' in metric_data:
                    for value_entry in metric_data['values']:
                        row = {
                            'metric': metric,
                            'date': value_entry.get('end_time', value_entry.get('datetime', '')),
                            'value': value_entry.get('value', 0)
                        }
                        all_data.append(row)

        if not all_data:
            logger.warning("⚠️ No page insights data returned")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)

        # Parse date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date

        # Pivot to have metrics as columns
        df_pivot = df.pivot(index='date', columns='metric', values='value').reset_index()

        logger.info(f"✅ Successfully fetched page insights for {len(df_pivot)} days")
        return df_pivot

    except requests.RequestException as e:
        logger.error(f"❌ Error fetching page insights: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"❌ Unexpected error fetching page insights: {e}", exc_info=True)
        return pd.DataFrame()

def fetch_page_posts(limit=25):
    """
    Fetch recent Facebook Page posts.

    Args:
        limit: number of posts to fetch

    Returns:
        pandas.DataFrame with post data

    Official docs: https://developers.facebook.com/docs/graph-api/reference/page/posts
    """
    page_id = os.getenv("PAGE_ID")
    access_token = os.getenv("META_ACCESS_TOKEN")

    if not page_id or not access_token:
        logger.error("❌ PAGE_ID or META_ACCESS_TOKEN not configured")
        return pd.DataFrame()

    try:
        api_version = "v18.0"
        url = f"https://graph.facebook.com/{api_version}/{page_id}/posts"
        params = {
            'fields': 'id,created_time,message,story,type,permalink_url',
            'limit': limit,
            'access_token': access_token
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        if 'data' not in data or not data['data']:
            logger.warning("⚠️ No posts found")
            return pd.DataFrame()

        df = pd.DataFrame(data['data'])

        # Parse created_time
        if 'created_time' in df.columns:
            df['created_time'] = pd.to_datetime(df['created_time'])

        logger.info(f"✅ Successfully fetched {len(df)} posts")
        return df

    except requests.RequestException as e:
        logger.error(f"❌ Error fetching page posts: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"❌ Unexpected error fetching page posts: {e}", exc_info=True)
        return pd.DataFrame()

def get_organic_insights(date_preset=None, since=None, until=None, metrics=None):
    """
    Wrapper to fetch organic page insights with preset date ranges.

    Args:
        date_preset: 'yesterday', 'last_7d', 'last_30d', etc.
        since/until: Custom date range (YYYY-MM-DD format)
        metrics: List of metrics to fetch

    Returns:
        pandas.DataFrame with organic insights data

    Official docs: https://developers.facebook.com/docs/graph-api/reference/page/insights/
    """
    if metrics is None:
        metrics = [
            "page_impressions_organic", "page_impressions_paid", 
            "page_engaged_users", "page_post_engagements",
            "page_fans", "page_fan_adds", "page_fan_removes"
        ]

    # Handle date presets
    if date_preset:
        today = datetime.now().date()
        if date_preset == "yesterday":
            dt = today - timedelta(days=1)
            since = until = dt.strftime("%Y-%m-%d")
        elif date_preset == "last_7d":
            until_dt = today - timedelta(days=1)
            since = (until_dt - timedelta(days=6)).strftime("%Y-%m-%d")
            until = until_dt.strftime("%Y-%m-%d")
        elif date_preset == "last_30d":
            until_dt = today - timedelta(days=1)
            since = (until_dt - timedelta(days=29)).strftime("%Y-%m-%d")
            until = until_dt.strftime("%Y-%m-%d")
        else:
            # Default to yesterday for unknown presets
            dt = today - timedelta(days=1)
            since = until = dt.strftime("%Y-%m-%d")

    if not since or not until:
        logger.error("❌ get_organic_insights: must supply since and until or valid date_preset")
        return pd.DataFrame()

    logger.info(f"📊 Fetching organic insights for {date_preset or f'{since} to {until}'}")
    return fetch_page_insights(metrics=metrics, since=since, until=until)

def get_organic_performance_summary(days=7):
    """
    Get a summary of organic performance for Facebook Page.

    Args:
        days: Number of days to look back

    Returns:
        dict with organic performance summary
    """
    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    # Fetch page insights
    page_insights = fetch_page_insights(
        since=start_date.strftime('%Y-%m-%d'),
        until=end_date.strftime('%Y-%m-%d')
    )

    # Fetch recent posts
    posts = fetch_page_posts(limit=10)

    summary = {
        'page_insights': page_insights.to_dict('records') if not page_insights.empty else [],
        'recent_posts': posts.to_dict('records') if not posts.empty else [],
        'posts_count': len(posts) if not posts.empty else 0
    }

    # Calculate averages if data exists
    if not page_insights.empty:
        numeric_columns = page_insights.select_dtypes(include=['number']).columns
        summary['page_averages'] = page_insights[numeric_columns].mean().to_dict()

    return summary