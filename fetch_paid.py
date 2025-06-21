"""
Enhanced fetch paid advertising data from Facebook Marketing API.
Uses facebook_business SDK for robust, paginated insights retrieval.

Updated with cleaned field lists, improved creative handling, and better error management.
"""

import os
import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from config import config
import json

# Import optimized API helpers
from api_helpers import safe_api_call, batch_facebook_requests, get_api_stats, sdk_call_with_backoff

# Import fb_client for API access
from fb_client import fb_client

# Facebook Business SDK imports
try:
    from facebook_business.adobjects.campaign import Campaign
    from facebook_business.adobjects.ad import Ad
    from facebook_business.adobjects.adcreative import AdCreative
    from facebook_business.adobjects.adaccount import AdAccount
    from facebook_business.api import FacebookRequestError
    CREATIVE_SDK_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Creative SDK imports not available: {e}")
    CREATIVE_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)

# Valid ad insight fields - removed all unsupported fields based on API docs
# Reference: https://developers.facebook.com/docs/marketing-api/reference/ads-insights/
VALID_AD_INSIGHT_FIELDS = [
    "ad_id", "ad_name", "adset_id", "adset_name", "campaign_id", "campaign_name",
    "impressions", "clicks", "spend", "reach", "frequency", "ctr", "cpc", "cpm",
    "date_start", "date_stop", "account_id", "account_name"
]

# Valid Ads Insights fields by level
VALID_INSIGHT_FIELDS = {
    "campaign": [
        "campaign_id", "campaign_name", "objective",
        "impressions", "clicks", "spend", "reach", "frequency", "ctr", "cpc", "cpm",
        "date_start", "date_stop", "account_id", "account_name"
    ],
    "adset": [
        "adset_id", "adset_name", "campaign_id", "campaign_name", "objective",
        "impressions", "clicks", "spend", "reach", "frequency", "ctr", "cpc", "cpm",
        "date_start", "date_stop", "account_id"
    ],
    "ad": VALID_AD_INSIGHT_FIELDS
}

def get_account_with_retries() -> Optional[AdAccount]:
    """Get Facebook Ad Account with safe API call."""
    if not hasattr(fb_client, "account") or fb_client.account is None:
        logger.error("❌ Facebook client not initialized")
        return None
    return fb_client.account

def safe_fetch_insights(level: str, fields: List[str], params: Dict) -> pd.DataFrame:
    """
    Safe insights fetcher with proper error handling and field validation.

    Args:
        level: 'campaign', 'adset', or 'ad'
        fields: List of valid insight fields
        params: API parameters

    Returns:
        DataFrame with insights data
    """
    account = get_account_with_retries()
    if account is None:
        return pd.DataFrame()

    # Validate fields against known valid fields
    valid_fields = VALID_INSIGHT_FIELDS.get(level, [])
    validated_fields = [f for f in fields if f in valid_fields]

    if len(validated_fields) != len(fields):
        invalid_fields = set(fields) - set(validated_fields)
        logger.warning(f"⚠️ Removed invalid {level} insight fields: {invalid_fields}")

    # Set parameters
    params["level"] = level

    def api_call():
        return account.get_insights(fields=validated_fields, params=params)

    # Create cache-friendly endpoint identifier
    date_range = params.get("date_preset", f"{params.get('time_range', {}).get('since', '')}_to_{params.get('time_range', {}).get('until', '')}")
    endpoint = f"insights_{level}_{date_range}"

    # Use safe API call with caching
    insights_data = safe_api_call(
        api_call,
        endpoint,
        params,
        use_cache=True,
        cache_ttl_hours=2
    )

    if not insights_data:
        logger.warning("⚠️ No insights data returned")
        return pd.DataFrame()

    # Process results
    all_data = []
    try:
        if isinstance(insights_data, list):
            all_data = insights_data
        elif isinstance(insights_data, dict):
            all_data = [insights_data]
        else:
            logger.warning(f"Unexpected insights data type: {type(insights_data)}")
            return pd.DataFrame()

        logger.info(f"✅ Successfully fetched {len(all_data)} {level} insight records")

    except Exception as e:
        logger.error(f"❌ Error processing insights data: {e}", exc_info=True)
        return pd.DataFrame()

    if not all_data:
        return pd.DataFrame()

    return pd.DataFrame(all_data)

def get_campaign_performance_optimized(
    date_preset: str = "last_7d",
    since: str = None,
    until: str = None,
    extra_fields: List[str] = None,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Optimized campaign performance fetcher with corrected fields.

    Args:
        date_preset: 'yesterday', 'last_7d', 'last_30d', etc.
        since/until: Custom date range
        extra_fields: Additional fields to include
        force_refresh: Force refresh cache

    Returns:
        DataFrame with campaign performance data
    """
    logger.info(f"📊 Fetching campaign performance for {date_preset or f'{since} to {until}'}")

    # Use validated campaign fields
    fields = VALID_INSIGHT_FIELDS["campaign"].copy()

    if extra_fields:
        # Only add extra fields that are in the valid list
        valid_extra = [f for f in extra_fields if f in VALID_INSIGHT_FIELDS["campaign"]]
        fields.extend([f for f in valid_extra if f not in fields])

    # Build params
    params = {}
    if date_preset:
        params["date_preset"] = date_preset
    elif since and until:
        params["time_range"] = {"since": since, "until": until}
    else:
        params["date_preset"] = "yesterday"

    return safe_fetch_insights("campaign", fields, params)

def fetch_creatives_for_ads(ad_ids: List[str]) -> Dict[str, Dict]:
    """
    Given a list of ad IDs, batch-fetch their creative details.
    Returns a dict: {ad_id: {creative fields...}, ...}
    """
    if not CREATIVE_SDK_AVAILABLE or not ad_ids:
        return {}

    creatives_map = {}
    BATCH_SIZE = 50

    for i in range(0, len(ad_ids), BATCH_SIZE):
        batch_ids = ad_ids[i:i + BATCH_SIZE]

        # Fetch creatives individually for each ad in the batch
        for ad_id in batch_ids:
            try:
                def get_ad_creative():
                    ad = Ad(ad_id)
                    creative_data = ad.api_get(fields=['creative'])
                    # Handle different response types
                    if isinstance(creative_data, dict):
                        return creative_data
                    elif hasattr(creative_data, 'export_all_data'):
                        return creative_data.export_all_data()
                    else:
                        return {'creative': {'id': str(creative_data)}}

                ad_info = safe_api_call(
                    get_ad_creative,
                    f"ad_{ad_id}",
                    {},
                    cache_ttl_hours=24
                )

                if ad_info and isinstance(ad_info, dict) and 'creative' in ad_info:
                    creative_info = ad_info['creative']

                    # Handle different creative response types
                    creative_id = None
                    if isinstance(creative_info, dict):
                        creative_id = creative_info.get('id')
                    elif isinstance(creative_info, str):
                        creative_id = creative_info
                    else:
                        creative_id = str(creative_info)

                    if creative_id:
                        # Fetch the actual creative details
                        def get_creative_details():
                            try:
                                creative = AdCreative(creative_id)
                                creative_details = creative.api_get(fields=[
                                    'id', 'name', 'body', 'title', 'image_url', 
                                    'thumbnail_url', 'object_url'
                                ])

                                # Handle response type
                                if isinstance(creative_details, dict):
                                    return creative_details
                                elif hasattr(creative_details, 'export_all_data'):
                                    return creative_details.export_all_data()
                                else:
                                    return {'id': creative_id, 'name': f'Creative {creative_id}'}
                            except Exception as e:
                                logger.debug(f"Creative details fetch failed for {creative_id}: {e}")
                                return {'id': creative_id, 'name': f'Creative {creative_id}'}

                        creative_details = safe_api_call(
                            get_creative_details,
                            f"creative_{creative_id}",
                            {},
                            cache_ttl_hours=24
                        )

                        if creative_details and isinstance(creative_details, dict):
                            creatives_map[ad_id] = creative_details
                        else:
                            creatives_map[ad_id] = {'id': creative_id, 'name': f'Creative {creative_id}'}
                    else:
                        creatives_map[ad_id] = {}
                else:
                    creatives_map[ad_id] = {}

            except Exception as e:
                logger.warning(f"Failed to fetch creative for ad {ad_id}: {e}")
                creatives_map[ad_id] = {}

    logger.info(f"✅ Fetched creative details for {len(creatives_map)}/{len(ad_ids)} ads")
    return creatives_map

def get_ad_performance_with_creatives(
    campaign_ids: List[str] = None,
    date_preset: str = "last_7d",
    include_creatives: bool = True,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Ad-level performance with creative data fetched separately.

    Args:
        campaign_ids: Filter by specific campaign IDs
        date_preset: Date range preset
        include_creatives: Include creative preview data
        force_refresh: Force refresh cache

    Returns:
        DataFrame with ad performance and creative data
    """
    # Build filtering
    params = {}
    if campaign_ids:
        params["filtering"] = [{'field': 'campaign.id', 'operator': 'IN', 'value': campaign_ids}]

    if date_preset:
        params["date_preset"] = date_preset
    else:
        params["date_preset"] = "last_7d"

    # Fetch ad insights with validated fields
    fields = VALID_INSIGHT_FIELDS["ad"].copy()
    df_ads = safe_fetch_insights("ad", fields, params)

    # Enrich with creative data if requested and we have ads
    if include_creatives and not df_ads.empty and 'ad_id' in df_ads.columns:
        unique_ad_ids = df_ads['ad_id'].unique().tolist()
        creative_data = fetch_creatives_for_ads(unique_ad_ids)

        if creative_data:
            # Merge creative data with performance data
            logger.info(f"Enriching {len(df_ads)} rows with creative data")
            creative_rows = []

            for _, row in df_ads.iterrows():
                ad_id = row['ad_id']
                creative = creative_data.get(ad_id, {})

                enriched_row = row.to_dict()
                enriched_row.update({
                    'creative_id': creative.get('id'),
                    'creative_name': creative.get('name'),
                    'creative_body': creative.get('body'),
                    'creative_title': creative.get('title'),
                    'creative_image_url': creative.get('image_url'),
                    'creative_thumbnail_url': creative.get('thumbnail_url'),
                    'creative_object_url': creative.get('object_url')
                })
                creative_rows.append(enriched_row)

            df_ads = pd.DataFrame(creative_rows)

    return df_ads

def get_paid_insights(
    date_preset: str = "last_7d",
    since: str = None,
    until: str = None,
    include_creatives: bool = True,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Main function to get paid campaign insights with corrected API calls.

    Args:
        date_preset: Date range preset
        since/until: Custom date range
        include_creatives: Include creative preview data
        force_refresh: Force refresh cache

    Returns:
        DataFrame with paid campaign performance and creative data
    """
    try:
        logger.info(f"📊 Fetching paid insights for {date_preset or f'{since} to {until}'}")

        if include_creatives:
            # Get ad-level data with creatives for richer insights
            return get_ad_performance_with_creatives(
                date_preset=date_preset,
                include_creatives=True,
                force_refresh=force_refresh
            )
        else:
            # Get campaign-level data only for performance
            return get_campaign_performance_optimized(
                date_preset=date_preset,
                since=since,
                until=until,
                force_refresh=force_refresh
            )

    except Exception as e:
        logger.error(f"❌ Error fetching paid insights: {e}", exc_info=True)
        return pd.DataFrame()

def get_campaign_performance_with_creatives(
    date_preset: str = "last_7d",
    include_creatives: bool = True,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Main function for campaign performance with creative data.
    """
    return get_ad_performance_with_creatives(
        date_preset=date_preset,
        include_creatives=include_creatives,
        force_refresh=force_refresh
    )

def get_campaign_performance_summary(
    date_preset: str = "last_7d",
    campaign_ids: List[str] = None,
    force_refresh: bool = False
) -> Dict:
    """
    Campaign performance summary with corrected metrics.
    """
    try:
        params = {}
        if campaign_ids:
            params["filtering"] = [{'field': 'campaign.id', 'operator': 'IN', 'value': campaign_ids}]

        if date_preset:
            params["date_preset"] = date_preset

        performance_data = safe_fetch_insights("campaign", VALID_INSIGHT_FIELDS["campaign"], params)

        if performance_data.empty:
            return {
                'total_campaigns': 0,
                'total_spend': 0,
                'total_impressions': 0,
                'total_clicks': 0,
                'average_ctr': 0,
                'average_cpc': 0,
                'api_stats': get_api_stats()
            }

        # Calculate summary metrics with proper type conversion
        total_spend = pd.to_numeric(performance_data['spend'], errors='coerce').fillna(0).sum()
        total_impressions = pd.to_numeric(performance_data['impressions'], errors='coerce').fillna(0).sum()
        total_clicks = pd.to_numeric(performance_data['clicks'], errors='coerce').fillna(0).sum()

        average_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        average_cpc = (total_spend / total_clicks) if total_clicks > 0 else 0

        summary = {
            'total_campaigns': len(performance_data),
            'total_spend': total_spend,
            'total_impressions': total_impressions,
            'total_clicks': total_clicks,
            'average_ctr': round(average_ctr, 2),
            'average_cpc': round(average_cpc, 2),
            'date_range': date_preset,
            'api_stats': get_api_stats()
        }

        logger.info(f"📈 Generated performance summary: {summary}")
        return summary

    except Exception as e:
        logger.error(f"❌ Error generating performance summary: {e}", exc_info=True)
        return {'api_stats': get_api_stats()}

def compute_paid_kpis(df: pd.DataFrame) -> Dict:
    """
    Compute key performance indicators (KPIs) from a DataFrame of paid ad insights.
    Args:
        df: DataFrame containing ad insights data.

    Returns:
        A dictionary containing computed KPIs.
    """
    # Initialize default values for KPIs
    total_spend = 0
    total_impressions = 0
    total_clicks = 0
    total_reach = 0

    if df.empty:
        return {
            'total_spend': total_spend,
            'total_impressions': total_impressions,
            'total_clicks': total_clicks,
            'total_reach': total_reach
        }

    # Aggregate metrics, handling potential missing columns and data types
    try:
        total_spend = df['spend'].astype(float).sum() if 'spend' in df else 0
        total_impressions = df['impressions'].astype(int).sum() if 'impressions' in df else 0
        total_clicks = df['clicks'].astype(int).sum() if 'clicks' in df else 0
        total_reach = df['reach'].astype(int).sum() if 'reach' in df else 0
    except (KeyError, ValueError) as e:
        logger.error(f"Error converting columns to numeric: {e}")
        return {
            'total_spend': 0,
            'total_impressions': 0,
            'total_clicks': 0,
            'total_reach': 0
        }

    # Compute additional metrics
    ctr = (total_clicks / total_impressions) * 100 if total_impressions > 0 else 0
    cpc = total_spend / total_clicks if total_clicks > 0 else 0
    cpm = (total_spend / total_impressions) * 1000 if total_impressions > 0 else 0

    kpis = {
        'total_spend': round(total_spend, 2),
        'total_impressions': total_impressions,
        'total_clicks': total_clicks,
        'total_reach': total_reach,
        'ctr': round(ctr, 2),
        'cpc': round(cpc, 2),
        'cpm': round(cpm, 2)
    }

    logger.info(f"Computed KPIs: {kpis}")
    return kpis

# Backward compatibility aliases
def get_campaign_insights(*args, **kwargs):
    """Alias for get_paid_insights"""
    return get_paid_insights(*args, **kwargs)

if __name__ == "__main__":
    # Test corrected paid insights
    logger.info("🧪 Testing corrected paid insights...")
    try:
        df_paid = get_paid_insights(date_preset="last_7d", force_refresh=True)
        print(f"Paid insights: {len(df_paid)} records")
        print(f"Columns: {df_paid.columns.tolist()}")

        if not df_paid.empty:
            # Check for creative data
            creative_cols = [col for col in df_paid.columns if 'creative' in col]
            print(f"Creative columns: {creative_cols}")

        # Print API usage stats
        stats = get_api_stats()
        print(f"API Usage Stats: {stats}")

    except Exception as e:
        print(f"Test failed: {e}")
        logger.error(f"Test error: {e}", exc_info=True)