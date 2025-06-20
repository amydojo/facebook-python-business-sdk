"""
Facebook Ads API integration for fetching paid campaign performance data.
Official docs: https://developers.facebook.com/docs/marketing-api/insights/
"""
import logging
import pandas as pd
from datetime import datetime, timedelta
from fb_client import fb_client

logger = logging.getLogger(__name__)

def fetch_ad_insights(level="campaign", fields=None, date_preset=None, since=None, until=None, filtering=None, breakdowns=None):
    """
    Generic fetcher for paid insights via Facebook Business SDK.
    Official docs: https://developers.facebook.com/docs/marketing-api/insights/

    Args:
        level: 'campaign', 'adset', or 'ad'
        fields: List of metrics to fetch
        date_preset: 'yesterday', 'last_7d', 'last_30d', etc.
        since/until: Custom date range (YYYY-MM-DD format)
        filtering: List of filters
        breakdowns: List of breakdowns

    Returns:
        pandas.DataFrame with insights data
    """
    account = getattr(fb_client, "account", None)
    if account is None:
        logger.error("❌ fetch_ad_insights: Facebook client not initialized")
        return pd.DataFrame()

    # Default fields if none provided
    if fields is None:
        if level == "campaign":
            fields = ["campaign_id", "campaign_name", "impressions", "clicks", "spend", "ctr", "cpc", "cpm"]
        elif level == "adset":
            fields = ["adset_id", "adset_name", "campaign_id", "impressions", "clicks", "spend", "ctr", "cpc"]
        else:  # ad level
            fields = ["ad_id", "ad_name", "adset_id", "campaign_id", "impressions", "clicks", "spend", "ctr", "cpc"]

    # Build params
    params = {"level": level}

    if date_preset:
        params["date_preset"] = date_preset
    elif since and until:
        params["time_range"] = {"since": since, "until": until}
    else:
        # Default to yesterday
        params["date_preset"] = "yesterday"

    if filtering:
        params["filtering"] = filtering
    if breakdowns:
        params["breakdowns"] = breakdowns

    all_data = []
    try:
        logger.info(f"🔍 Fetching {level} insights with params: {params}")

        # Get insights from Facebook API
        insights = account.get_insights(fields=fields, params=params)

        # Process all pages of results
        while insights:
            for entry in insights:
                all_data.append(entry.export_all_data())
            try:
                insights = insights.next_page()
            except Exception:
                # No more pages
                break

        logger.info(f"✅ Successfully fetched {len(all_data)} {level} insight records")

    except Exception as e:
        logger.error(f"❌ fetch_ad_insights error: {e}", exc_info=True)
        return pd.DataFrame()

    if not all_data:
        logger.warning("⚠️ No insights data returned from Facebook API")
        return pd.DataFrame()

    return pd.DataFrame(all_data)

def get_campaign_performance(date_preset="last_7d", since=None, until=None, extra_fields=None):
    """
    Wrapper to fetch campaign-level performance data.

    Args:
        date_preset: 'yesterday', 'last_7d', 'last_30d', etc.
        since/until: Custom date range (YYYY-MM-DD format)
        extra_fields: Additional fields to include

    Returns:
        pandas.DataFrame with campaign performance data
    """
    logger.info(f"📊 Fetching campaign performance for {date_preset or f'{since} to {until}'}")

    base_fields = [
        "campaign_id", "campaign_name", "impressions", "clicks", "spend", 
        "reach", "frequency", "ctr", "cpc", "cpm", "date_start", "date_stop"
    ]

    if extra_fields:
        for field in extra_fields:
            if field not in base_fields:
                base_fields.append(field)

    return fetch_ad_insights(
        level="campaign", 
        fields=base_fields, 
        date_preset=date_preset, 
        since=since, 
        until=until
    )

def get_campaign_performance_summary(campaign_ids=None, date_preset='last_7d'):
    """
    Get summarized campaign performance data.

    Args:
        campaign_ids: List of specific campaign IDs (optional)
        date_preset: Date range preset

    Returns:
        dict: Summary statistics
    """
    try:
        performance_data = get_campaign_performance(date_preset=date_preset)

        if performance_data.empty:
            return {
                'total_campaigns': 0,
                'total_spend': 0,
                'total_impressions': 0,
                'total_clicks': 0,
                'average_ctr': 0,
                'average_cpc': 0
            }

        # Filter by campaign IDs if provided
        if campaign_ids:
            performance_data = performance_data[
                performance_data['campaign_id'].isin(campaign_ids)
            ]

        # Calculate summary metrics
        total_spend = performance_data['spend'].astype(float).sum()
        total_impressions = performance_data['impressions'].astype(int).sum()
        total_clicks = performance_data['clicks'].astype(int).sum()

        average_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        average_cpc = (total_spend / total_clicks) if total_clicks > 0 else 0

        summary = {
            'total_campaigns': len(performance_data),
            'total_spend': total_spend,
            'total_impressions': total_impressions,
            'total_clicks': total_clicks,
            'average_ctr': round(average_ctr, 2),
            'average_cpc': round(average_cpc, 2),
            'date_range': date_preset
        }

        logger.info(f"📈 Generated performance summary: {summary}")
        return summary

    except Exception as e:
        logger.error(f"❌ Error generating campaign performance summary: {e}", exc_info=True)
        return {}

def get_ad_performance(campaign_id=None, date_preset='last_7d'):
    """
    Get ad-level performance data.

    Args:
        campaign_id: Specific campaign ID to filter by
        date_preset: Date range preset

    Returns:
        pandas.DataFrame: Ad performance data
    """
    filtering = None
    if campaign_id:
        filtering = [{'field': 'campaign.id', 'operator': 'IN', 'value': [campaign_id]}]

    return fetch_ad_insights(
        level="ad",
        fields=[
            "ad_id", "ad_name", "campaign_id", "campaign_name", 
            "adset_id", "adset_name", "impressions", "clicks", 
            "spend", "ctr", "cpc", "cpm"
        ],
        date_preset=date_preset,
        filtering=filtering
    )

def get_real_time_insights(campaign_ids=None):
    """
    Get real-time campaign insights (today's data).

    Args:
        campaign_ids: List of campaign IDs to check

    Returns:
        pandas.DataFrame: Real-time insights data
    """
    filtering = None
    if campaign_ids:
        filtering = [{'field': 'campaign.id', 'operator': 'IN', 'value': campaign_ids}]

    return fetch_ad_insights(
        level="campaign",
        fields=[
            "campaign_id", "campaign_name", "impressions", 
            "clicks", "spend", "date_start", "date_stop"
        ],
        date_preset="today",
        filtering=filtering
    )