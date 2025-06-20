"""
Facebook Ads API integration for fetching paid campaign performance data.
Official docs: https://developers.facebook.com/docs/marketing-api/insights/
"""
import os
import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from config import config

# Import fb_client for API access
from fb_client import fb_client

# Facebook Business SDK imports for creative previews
# Official docs: https://developers.facebook.com/docs/marketing-api/reference/ad-creative/
try:
    from facebook_business.adobjects.campaign import Campaign
    from facebook_business.adobjects.ad import Ad
    from facebook_business.adobjects.adcreative import AdCreative
    CREATIVE_SDK_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Creative SDK imports not available: {e}")
    CREATIVE_SDK_AVAILABLE = False

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

def enrich_with_creatives(df_campaigns: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich campaign performance data with ad creative previews.

    Official docs: https://developers.facebook.com/docs/marketing-api/reference/ad-creative/

    Args:
        df_campaigns: DataFrame with campaign performance metrics

    Returns:
        DataFrame with creative preview fields added
    """
    if not CREATIVE_SDK_AVAILABLE:
        logger.warning("Creative SDK not available, returning campaigns without creative data")
        return df_campaigns

    if df_campaigns.empty:
        return pd.DataFrame(columns=[
            "campaign_id", "campaign_name", "ad_id", "ad_name", "creative_id", "creative_name",
            "creative_body", "creative_title", "creative_image_url", "creative_thumbnail_url",
            "creative_object_url", "impressions", "clicks", "spend", "reach", "frequency", 
            "ctr", "cpc", "date_start", "date_stop"
        ])

    records = []

    for _, row in df_campaigns.iterrows():
        campaign_id = row.get('campaign_id')
        campaign_name = row.get('campaign_name')

        if not campaign_id:
            continue

        try:
            # Fetch ads for this campaign
            # Official docs: https://developers.facebook.com/docs/marketing-api/reference/campaign/ads/
            ads = Campaign(campaign_id).get_ads(fields=[
                Ad.Field.id,
                Ad.Field.name,
                Ad.Field.creative
            ])

            for ad in ads:
                ad_id = ad.get(Ad.Field.id)
                ad_name = ad.get(Ad.Field.name)
                creative_id = None
                creative_name = creative_body = creative_title = None
                creative_image_url = creative_thumbnail_url = creative_object_url = None

                # Extract creative info
                creative_info = ad.get(Ad.Field.creative)
                if creative_info:
                    creative_id = creative_info.get('id')

                    if creative_id:
                        try:
                            # Fetch creative details with preview URLs
                            creative = AdCreative(creative_id).api_get(fields=[
                                AdCreative.Field.name,
                                AdCreative.Field.body,
                                AdCreative.Field.title,
                                AdCreative.Field.image_url,
                                AdCreative.Field.thumbnail_url,
                                AdCreative.Field.object_url
                            ])

                            creative_name = creative.get(AdCreative.Field.name)
                            creative_body = creative.get(AdCreative.Field.body)
                            creative_title = creative.get(AdCreative.Field.title)
                            creative_image_url = creative.get(AdCreative.Field.image_url)
                            creative_thumbnail_url = creative.get(AdCreative.Field.thumbnail_url)
                            creative_object_url = creative.get(AdCreative.Field.object_url)

                        except Exception as e:
                            logger.warning(f"Failed to fetch creative details for creative {creative_id}: {e}")

                # Merge with performance row
                record = {
                    "campaign_id": campaign_id,
                    "campaign_name": campaign_name,
                    "ad_id": ad_id,
                    "ad_name": ad_name,
                    "creative_id": creative_id,
                    "creative_name": creative_name,
                    "creative_body": creative_body,
                    "creative_title": creative_title,
                    "creative_image_url": creative_image_url,
                    "creative_thumbnail_url": creative_thumbnail_url,
                    "creative_object_url": creative_object_url,
                }

                # Copy performance metrics from row
                for col in ['impressions', 'clicks', 'spend', 'reach', 'frequency', 'ctr', 'cpc', 'cpm', 'date_start', 'date_stop']:
                    record[col] = row.get(col, 0)

                records.append(record)

        except Exception as e:
            logger.warning(f"Failed to fetch ads for campaign {campaign_id}: {e}")
            # Add campaign row without creative data
            record = {
                "campaign_id": campaign_id,
                "campaign_name": campaign_name,
                "ad_id": None,
                "ad_name": None,
                "creative_id": None,
                "creative_name": None,
                "creative_body": None,
                "creative_title": None,
                "creative_image_url": None,
                "creative_thumbnail_url": None,
                "creative_object_url": None,
            }
            for col in ['impressions', 'clicks', 'spend', 'reach', 'frequency', 'ctr', 'cpc', 'cpm', 'date_start', 'date_stop']:
                record[col] = row.get(col, 0)
            records.append(record)

    if not records:
        return pd.DataFrame(columns=[
            "campaign_id", "campaign_name", "ad_id", "ad_name", "creative_id", "creative_name",
            "creative_body", "creative_title", "creative_image_url", "creative_thumbnail_url", 
            "creative_object_url", "impressions", "clicks", "spend", "reach", "frequency",
            "ctr", "cpc", "cpm", "date_start", "date_stop"
        ])

    df_enriched = pd.DataFrame(records)
    logger.info(f"Enriched {len(df_enriched)} campaign records with creative data")
    return df_enriched

def get_campaign_performance_with_creatives(date_preset: str = "last_7d", include_creatives: bool = True) -> pd.DataFrame:
    """
    Get campaign performance data enriched with creative previews.

    Args:
        date_preset: Date range preset
        include_creatives: Whether to fetch creative preview data

    Returns:
        DataFrame with campaign performance and creative data
    """
    # Get base campaign performance
    df_campaigns = get_campaign_performance(date_preset=date_preset)

    if include_creatives and not df_campaigns.empty:
        return enrich_with_creatives(df_campaigns)

    return df_campaigns

def get_campaign_performance_summary(date_preset: str = "last_7d", campaign_ids: List[str] = None) -> Dict:
    """
    Get summarized campaign performance data.

    Args:
        date_preset: Date range preset
        campaign_ids: List of specific campaign IDs (optional)

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

if __name__ == "__main__":
    # Set env vars for testing
    import os
    # os.environ["PAGE_ACCESS_TOKEN"] = "<PAGE_ACCESS_TOKEN>"
    # os.environ["IG_USER_ID"] = "<IG_USER_ID>"
    # os.environ["AD_ACCOUNT_ID"] = "<AD_ACCOUNT_ID>"
    # os.environ["META_ACCESS_TOKEN"] = "<META_ACCESS_TOKEN>"
    # os.environ["META_APP_ID"] = "<META_APP_ID>"
    # os.environ["META_APP_SECRET"] = "<META_APP_SECRET>"
    
    # Test fb_client
    print("fb_client.account:", getattr(fb_client, "account", None))
    print("fb_client initialized:", fb_client.is_initialized())
    
    # Test paid fetch
    logger.info("🧪 Testing paid campaign fetch with creative previews...")
    try:
        df_paid = get_campaign_performance_with_creatives(date_preset="last_7d")
        print("Paid head:", df_paid.head() if not df_paid.empty else "Empty DataFrame")
        print("Paid cols:", df_paid.columns.tolist())
        
        if not df_paid.empty:
            # Check for preview URLs
            has_images = df_paid['creative_image_url'].notna().sum() if 'creative_image_url' in df_paid.columns else 0
            has_thumbnails = df_paid['creative_thumbnail_url'].notna().sum() if 'creative_thumbnail_url' in df_paid.columns else 0
            print(f"Preview URLs found: {has_images} images, {has_thumbnails} thumbnails")
    except Exception as e:
        print(f"Paid test failed: {e}")
        logger.error(f"Paid test error: {e}", exc_info=True)