"""
Fetch organic insights from Facebook Pages and Instagram.
References:
- Page Insights: https://developers.facebook.com/docs/graph-api/reference/page/insights/
- Instagram Graph API: https://developers.facebook.com/docs/instagram-api/
"""
import logging
import pandas as pd
import requests
from datetime import datetime, timedelta
from config import config
from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects import Page

logger = logging.getLogger(__name__)

def fetch_page_insights(metrics=None, since=None, until=None, period='day'):
    """
    Fetch Facebook Page insights.

    Args:
        metrics: list of metrics to fetch
        since: start date (YYYY-MM-DD)
        until: end date (YYYY-MM-DD)
        period: 'day', 'week', 'days_28'

    Returns:
        pandas.DataFrame with page insights

    Reference: GET /{PAGE_ID}/insights?metric=...&period=...&since=...&until=...
    """
    if not config.PAGE_ID or not config.META_ACCESS_TOKEN:
        logger.error("PAGE_ID or META_ACCESS_TOKEN not configured")
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

        # Fetch each metric separately (API limitation)
        for metric in metrics:
            logger.info(f"Fetching page metric: {metric}")

            url = f"https://graph.facebook.com/{config.GRAPH_API_VERSION}/{config.PAGE_ID}/insights"
            params = {
                'metric': metric,
                'period': period,
                'since': since,
                'until': until,
                'access_token': config.META_ACCESS_TOKEN
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
            logger.warning("No page insights data returned")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)

        # Parse date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date

        # Pivot to have metrics as columns
        df_pivot = df.pivot(index='date', columns='metric', values='value').reset_index()

        logger.info(f"Successfully fetched page insights for {len(df_pivot)} days")
        return df_pivot

    except requests.RequestException as e:
        logger.error(f"Error fetching page insights: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error fetching page insights: {e}")
        return pd.DataFrame()

def fetch_page_posts(limit=25):
    """
    Fetch recent Facebook Page posts.

    Args:
        limit: number of posts to fetch

    Returns:
        pandas.DataFrame with post data

    Reference: GET /{PAGE_ID}/posts?fields=id,created_time,message,story
    """
    if not config.PAGE_ID or not config.META_ACCESS_TOKEN:
        logger.error("PAGE_ID or META_ACCESS_TOKEN not configured")
        return pd.DataFrame()

    try:
        url = f"https://graph.facebook.com/{config.GRAPH_API_VERSION}/{config.PAGE_ID}/posts"
        params = {
            'fields': 'id,created_time,message,story,type,permalink_url',
            'limit': limit,
            'access_token': config.META_ACCESS_TOKEN
        }

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()

        if 'data' not in data or not data['data']:
            logger.warning("No posts found")
            return pd.DataFrame()

        df = pd.DataFrame(data['data'])

        # Parse created_time
        if 'created_time' in df.columns:
            df['created_time'] = pd.to_datetime(df['created_time'])

        logger.info(f"Successfully fetched {len(df)} posts")
        return df

    except requests.RequestException as e:
        logger.error(f"Error fetching page posts: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error fetching page posts: {e}")
        return pd.DataFrame()

def fetch_post_insights(post_id, metrics=None, since=None, until=None, period='lifetime'):
    """
    Fetch insights for a specific post.

    Args:
        post_id: Facebook post ID
        metrics: list of metrics to fetch
        since: start date (YYYY-MM-DD)
        until: end date (YYYY-MM-DD)
        period: 'lifetime', 'day'

    Returns:
        pandas.DataFrame with post insights

    Reference: GET /{POST_ID}/insights?metric=...
    """
    if not config.META_ACCESS_TOKEN:
        logger.error("META_ACCESS_TOKEN not configured")
        return pd.DataFrame()

    # Default metrics for posts
    if metrics is None:
        metrics = [
            'post_impressions', 'post_reach', 'post_engaged_users',
            'post_clicks', 'post_reactions_like_total',
            'post_reactions_love_total', 'post_reactions_wow_total',
            'post_reactions_haha_total', 'post_reactions_sorry_total',
            'post_reactions_anger_total'
        ]

    try:
        all_data = []

        for metric in metrics:
            url = f"https://graph.facebook.com/{config.GRAPH_API_VERSION}/{post_id}/insights"
            params = {
                'metric': metric,
                'period': period,
                'access_token': config.META_ACCESS_TOKEN
            }

            # Add date range for non-lifetime periods
            if period != 'lifetime' and since and until:
                params['since'] = since
                params['until'] = until

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if 'data' in data and data['data']:
                metric_data = data['data'][0]

                if 'values' in metric_data:
                    for value_entry in metric_data['values']:
                        row = {
                            'post_id': post_id,
                            'metric': metric,
                            'value': value_entry.get('value', 0)
                        }
                        all_data.append(row)

        if not all_data:
            logger.warning(f"No insights data for post {post_id}")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)

        # Pivot to have metrics as columns
        df_pivot = df.pivot(index='post_id', columns='metric', values='value').reset_index()

        return df_pivot

    except requests.RequestException as e:
        logger.error(f"Error fetching post insights for {post_id}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error fetching post insights for {post_id}: {e}")
        return pd.DataFrame()

def fetch_ig_media_insights(media_id, metrics=None, period='lifetime'):
    """
    Fetch Instagram media insights (organic only).

    Args:
        media_id: Instagram media ID
        metrics: list of metrics to fetch
        period: 'lifetime', 'day'

    Returns:
        pandas.DataFrame with Instagram media insights

    Reference: GET /{MEDIA_ID}/insights?metric=... (Instagram Graph API)
    Note: For paid IG metrics, use Ads Insights API filtered by creative
    """
    if not config.META_ACCESS_TOKEN:
        logger.error("META_ACCESS_TOKEN not configured")
        return pd.DataFrame()

    # Default metrics for Instagram media
    if metrics is None:
        metrics = [
            'impressions', 'reach', 'engagement', 'saved',
            'video_views', 'likes', 'comments', 'shares'
        ]

    try:
        all_data = []

        for metric in metrics:
            url = f"https://graph.facebook.com/{config.GRAPH_API_VERSION}/{media_id}/insights"
            params = {
                'metric': metric,
                'period': period,
                'access_token': config.META_ACCESS_TOKEN
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if 'data' in data and data['data']:
                for metric_data in data['data']:
                    if 'values' in metric_data:
                        for value_entry in metric_data['values']:
                            row = {
                                'media_id': media_id,
                                'metric': metric_data.get('name', metric),
                                'value': value_entry.get('value', 0)
                            }
                            all_data.append(row)

        if not all_data:
            logger.warning(f"No insights data for Instagram media {media_id}")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)

        # Pivot to have metrics as columns
        df_pivot = df.pivot(index='media_id', columns='metric', values='value').reset_index()

        return df_pivot

    except requests.RequestException as e:
        logger.error(f"Error fetching Instagram media insights for {media_id}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error fetching Instagram media insights for {media_id}: {e}")
        return pd.DataFrame()

def get_organic_insights(days=7):
    """
    Get organic insights for Facebook Page and Instagram.

    Args:
        days: Number of days to look back

    Returns:
        dict with organic insights data
    """
    try:
        if not config.PAGE_ID or not config.META_ACCESS_TOKEN:
            logger.error("PAGE_ID or META_ACCESS_TOKEN not configured")
            return {
                'page_insights': [],
                'recent_posts': [],
                'posts_count': 0,
                'instagram_user_id': None,
                'error': 'PAGE_ID or META_ACCESS_TOKEN not configured',
                'date_range': {
                    'start': (datetime.now().date() - timedelta(days=days)).strftime('%Y-%m-%d'),
                    'end': datetime.now().date().strftime('%Y-%m-%d')
                }
            }

        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)

        logger.info(f"Fetching organic insights from {start_date} to {end_date}")

        # Initialize Facebook API
        FacebookAdsApi.init(
            app_id=config.META_APP_ID,
            app_secret=config.META_APP_SECRET,
            access_token=config.META_ACCESS_TOKEN
        )

        page = Page(config.PAGE_ID)

        # Get Instagram User ID if available
        instagram_user_id = None
        try:
            # Get Instagram accounts connected to this page
            instagram_accounts = page.get_instagram_accounts(
                fields=['id', 'username', 'name']
            )
            if instagram_accounts:
                instagram_account = instagram_accounts[0]  # Get the first connected account
                instagram_user_id = instagram_account.get('id')
                logger.info(f"Retrieved Instagram User ID: {instagram_user_id}")
            else:
                logger.info("No Instagram accounts found for this page")
        except Exception as e:
            logger.warning(f"Could not retrieve Instagram User ID: {e}")

        # Get page insights
        page_insights = []
        try:
            insights = page.get_insights(
                fields=[
                    'page_impressions',
                    'page_impressions_unique',
                    'page_post_engagements',
                    'page_fans_city',
                    'page_fans_country',
                    'page_content_activity'
                ],
                params={
                    'since': start_date.strftime('%Y-%m-%d'),
                    'until': end_date.strftime('%Y-%m-%d'),
                    'period': 'day'
                }
            )
            page_insights = [insight.export_all_data() for insight in insights]
            logger.info(f"Retrieved {len(page_insights)} page insights")
        except Exception as e:
            logger.error(f"Error fetching page insights: {e}")

        # Get recent posts
        recent_posts = []
        try:
            posts = page.get_posts(
                fields=[
                    'id',
                    'message',
                    'created_time',
                    'type',
                    'likes.summary(true)',
                    'comments.summary(true)',
                    'shares'
                ],
                params={
                    'since': start_date.strftime('%Y-%m-%d'),
                    'until': end_date.strftime('%Y-%m-%d'),
                    'limit': 50
                }
            )
            recent_posts = [post.export_all_data() for post in posts]
            logger.info(f"Retrieved {len(recent_posts)} recent posts")
        except Exception as e:
            logger.error(f"Error fetching recent posts: {e}")

        return {
            'page_insights': page_insights,
            'recent_posts': recent_posts,
            'posts_count': len(recent_posts),
            'instagram_user_id': instagram_user_id,
            'date_range': {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            }
        }

    except Exception as e:
        logger.error(f"Error getting organic insights: {e}")
        return {
            'page_insights': [],
            'recent_posts': [],
            'posts_count': 0,
            'instagram_user_id': None,
            'error': str(e),
            'date_range': {
                'start': (datetime.now().date() - timedelta(days=days)).strftime('%Y-%m-%d'),
                'end': datetime.now().date().strftime('%Y-%m-%d')
            }
        }

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
        'page_insights': page_insights,
        'recent_posts': posts,
        'posts_count': len(posts) if not posts.empty else 0
    }

    # Calculate averages if data exists
    if not page_insights.empty:
        numeric_columns = page_insights.select_dtypes(include=['number']).columns
        summary['page_averages'] = page_insights[numeric_columns].mean().to_dict()

    return summary