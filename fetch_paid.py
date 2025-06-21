"""
Enhanced fetch paid advertising data from Facebook Marketing API.
Uses facebook_business SDK for robust, paginated insights retrieval.

Official docs:
- Ads Insights API: https://developers.facebook.com/docs/marketing-api/insights/
- Batch Requests: https://developers.facebook.com/docs/marketing-api/best-practices/
- Error Handling: https://developers.facebook.com/docs/marketing-api/error-handling/

Updated with enhanced error handling, batch processing, and metric optimization
"""

# Valid ad insight fields to prevent 400 errors
VALID_AD_INSIGHT_FIELDS = {
    "ad_id", "ad_name", "adset_id", "adset_name", "campaign_id", "campaign_name",
    "impressions", "clicks", "spend", "reach", "frequency", "ctr", "cpc", "cpm",
    "unique_clicks", "unique_link_clicks", "cost_per_unique_click", 
    "date_start", "date_stop", "actions", "action_values", "conversions",
    "conversion_values", "cost_per_action_type", "video_30_sec_watched_actions",
    "video_p25_watched_actions", "video_p50_watched_actions", "video_p75_watched_actions",
    "video_p100_watched_actions", "video_play_actions", "outbound_clicks",
    "unique_outbound_clicks", "inline_link_clicks", "unique_inline_link_clicks"
}
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
try:
    from fb_client import validate_credentials
except ImportError:
    def validate_credentials():
        """Fallback validate_credentials if not available in fb_client"""
        return True

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

# Valid Ads Insights fields based on official Meta Marketing API docs
# https://developers.facebook.com/docs/marketing-api/reference/ads-insights/
VALID_INSIGHT_FIELDS = {
    "campaign": [
        "campaign_id", "campaign_name", "objective", "status", "budget_rebalance_flag",
        "impressions", "clicks", "spend", "reach", "frequency", "ctr", "cpc", "cpm", "cpp",
        "unique_clicks", "unique_link_clicks_ctr", "cost_per_unique_click", "social_spend",
        "date_start", "date_stop", "account_id", "account_name"
    ],
    "adset": [
        "adset_id", "adset_name", "campaign_id", "campaign_name", "status", "objective",
        "optimization_goal", "billing_event", "bid_amount", "budget_remaining", "daily_budget",
        "impressions", "clicks", "spend", "reach", "frequency", "ctr", "cpc", "cpm",
        "date_start", "date_stop", "account_id"
    ],
    "ad": [
        "ad_id", "ad_name", "adset_id", "adset_name", "campaign_id", "campaign_name", 
        "status", "impressions", "clicks", "spend", "reach", "frequency", "ctr", "cpc", "cpm",
        "unique_clicks", "unique_link_clicks_ctr", "cost_per_unique_click",
        "date_start", "date_stop", "account_id"
        # NOTE: creative fields cannot be included in insights - must fetch separately
    ]
}

def get_account_with_retries() -> Optional[AdAccount]:
    """Get Facebook Ad Account with safe API call."""
    if not hasattr(fb_client, "account") or fb_client.account is None:
        logger.error("❌ Facebook client not initialized")
        return None
    return fb_client.account

def fetch_creative_details(ad_ids: List[str]) -> Dict[str, Dict]:
    """
    Fetch creative details for multiple ads using separate API calls.

    Args:
        ad_ids: List of ad IDs to fetch creative details for

    Returns:
        Dict mapping ad_id to creative details
    """
    if not CREATIVE_SDK_AVAILABLE or not ad_ids:
        return {}

    creative_data = {}

    for ad_id in ad_ids[:50]:  # Limit to 50 ads for safety
        try:
            def get_ad_creative():
                ad = Ad(ad_id)
                return ad.api_get(fields=['creative'])

            ad_info = safe_api_call(
                get_ad_creative,
                f"ad_{ad_id}",
                {},
                cache_ttl_hours=24
            )

            if ad_info and 'creative' in ad_info:
                creative_id = ad_info['creative']['id']

                # Fetch the actual creative details
                def get_creative_details():
                    creative = AdCreative(creative_id)
                    return creative.api_get(fields=[
                        'id', 'name', 'body', 'title', 'image_url', 
                        'thumbnail_url', 'object_url', 'image_hash'
                    ])

                creative_details = safe_api_call(
                    get_creative_details,
                    f"creative_{creative_id}",
                    {},
                    cache_ttl_hours=24
                )

                if creative_details:
                    creative_dict = creative_details.export_all_data() if hasattr(creative_details, 'export_all_data') else creative_details
                    creative_data[ad_id] = creative_dict

        except Exception as e:
            logger.warning(f"Failed to fetch creative for ad {ad_id}: {e}")
            continue

    logger.info(f"✅ Fetched creative details for {len(creative_data)}/{len(ad_ids)} ads")
    return creative_data

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
        else:
            page_count = 0
            while insights_data:
                page_count += 1
                logger.info(f"📄 Processing insights page {page_count}")

                for entry in insights_data:
                    data = entry.export_all_data() if hasattr(entry, 'export_all_data') else entry
                    all_data.append(data)

                try:
                    insights_data = safe_api_call(
                        lambda: insights_data.next_page(),
                        f"{endpoint}_page_{page_count}",
                        {},
                        use_cache=False
                    )
                except Exception:
                    break

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

    # Fetch ad insights with validated fields (no creative fields)
    fields = VALID_INSIGHT_FIELDS["ad"].copy()
    df_ads = safe_fetch_insights("ad", fields, params)

    # Enrich with creative data if requested and we have ads
    if include_creatives and not df_ads.empty and 'ad_id' in df_ads.columns:
        unique_ad_ids = df_ads['ad_id'].unique().tolist()
        creative_data = fetch_creative_details(unique_ad_ids)

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

# Backward compatibility aliases
def get_campaign_insights(*args, **kwargs):
    """Alias for get_paid_insights"""
    return get_paid_insights(*args, **kwargs)

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

def fetch_creative_for_ad(ad_id: str) -> List[Dict]:
    """
    Fetch creative details for a specific ad ID.

    Args:
        ad_id: Ad ID to fetch creative for

    Returns:
        List of creative dictionaries
    """
    try:
        ad = Ad(ad_id)
        creatives = sdk_call_with_backoff(
            ad.get_ad_creatives,
            fields=[
                "id", "name", "body", "title", "image_url", 
                "thumbnail_url", "object_url", "call_to_action_type"
            ]
        )

        result = []
        for creative in creatives or []:
            try:
                if hasattr(creative, 'export_all_data'):
                    result.append(creative.export_all_data())
                elif isinstance(creative, dict):
                    result.append(creative)
                else:
                    logger.debug(f"Skipping unexpected creative type: {type(creative)}")
            except Exception as e:
                logger.warning(f"Failed to export creative data for ad {ad_id}: {e}")

        return result

    except Exception as e:
        logger.warning(f"Failed to fetch creative for ad {ad_id}: {e}")
        return []
    
def fetch_ad_insights_fields(account_id: str, level: str, fields: List[str], 
                                date_preset: Optional[str] = None, 
                                since: Optional[str] = None, 
                                until: Optional[str] = None,
                                params_extra: Optional[Dict[str, Any]] = None) -> List[Dict]:
    """
    Fetch ad insights using facebook_business SDK with enhanced error handling.

    Args:
        account_id: Ad account ID
        level: 'campaign', 'adset', or 'ad'
        fields: List of insight fields to fetch
        date_preset: Preset like 'last_7d', 'yesterday'
        since: Start date in YYYY-MM-DD format
        until: End date in YYYY-MM-DD format
        params_extra: Additional parameters

    Returns:
        List of insight records as dictionaries
    """
    if not validate_credentials():
        logger.error("❌ Invalid credentials for paid insights")
        return []

    # Filter out invalid fields to prevent 400 errors
    filtered_fields = [f for f in fields if f in VALID_AD_INSIGHT_FIELDS]
    invalid_fields = set(fields) - set(filtered_fields)

    if invalid_fields:
        logger.warning(f"⚠️ Removed invalid insight fields: {invalid_fields}")

    if not filtered_fields:
        logger.error("❌ No valid insight fields provided")
        return []

    # Ensure account_id has proper prefix
    account_id = account_id if account_id.startswith("act_") else f"act_{account_id}"

    try:
        account = AdAccount(account_id)

        # Build insight parameters
        insight_params = {
            "level": level,
            "fields": ",".join(filtered_fields)
        }

        # Add date range
        if date_preset:
            insight_params["date_preset"] = date_preset
        elif since and until:
            insight_params["time_range"] = json.dumps({
                "since": since,
                "until": until
            })

        # Add extra parameters
        if params_extra:
            insight_params.update(params_extra)

        logger.info(f"🎯 Fetching {level} insights for account {account_id}")
        logger.debug(f"📋 Insight parameters: {insight_params}")

        # Make SDK call with retry logic
        insights = sdk_call_with_backoff(
            account.get_insights,
            params=insight_params
        )

        if not insights:
            logger.warning(f"⚠️ No insights returned for account {account_id}")
            return []

        # Convert SDK objects to dictionaries - handle different response types
        results = []
        for insight in insights:
            try:
                if hasattr(insight, 'export_all_data'):
                    data = insight.export_all_data()
                    results.append(data)
                elif isinstance(insight, dict):
                    results.append(insight)
                else:
                    logger.debug(f"⚠️ Skipping unexpected insight type: {type(insight)}")

            except Exception as e:
                logger.warning(f"❌ Error exporting insight data: {e}")
                continue

        logger.info(f"✅ Successfully fetched {len(results)} {level} insight records")
        return results

    except FacebookRequestError as e:
        error_code = e.api_error_code()
        error_message = e.api_error_message()
        logger.error(f"❌ Facebook API error (code {error_code}): {error_message}")

        # Handle specific error cases
        if error_code == 17:  # User request limit reached
            logger.warning("⏳ Rate limit reached, consider implementing backoff")
        elif error_code == 190:  # Access token issues
            logger.error("🔑 Access token error - check token validity")
        elif error_code == 100:  # Invalid parameter
            logger.error(f"📋 Invalid parameters: {insight_params}")

        return []

    except Exception as e:
        logger.error(f"❌ Unexpected error fetching {level} insights: {e}", exc_info=True)
        return []
    
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
    total_conversions = 0

    if df.empty:
        return {
            'total_spend': total_spend,
            'total_impressions': total_impressions,
            'total_clicks': total_clicks,
            'total_reach': total_reach,
            'total_conversions': total_conversions
        }

    # Aggregate metrics, handling potential missing columns and data types
    try:
        total_spend = df['spend'].astype(float).sum() if 'spend' in df else 0
        total_impressions = df['impressions'].astype(int).sum() if 'impressions' in df else 0
        total_clicks = df['clicks'].astype(int).sum() if 'clicks' in df else 0
        total_reach = df['reach'].astype(int).sum() if 'reach' in df else 0
        total_conversions = df['conversions'].astype(int).sum() if 'conversions' in df else 0
    except KeyError as e:
        logger.error(f"Missing column in DataFrame: {e}")
        return {
            'total_spend': 0,
            'total_impressions': 0,
            'total_clicks': 0,
            'total_reach': 0,
            'total_conversions': 0
        }
    except ValueError as e:
        logger.error(f"Error converting column to numeric type: {e}")
        return {
            'total_spend': 0,
            'total_impressions': 0,
            'total_clicks': 0,
            'total_reach': 0,
            'total_conversions': 0
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
        'total_conversions': total_conversions,
        'ctr': round(ctr, 2),
        'cpc': round(cpc, 2),
        'cpm': round(cpm, 2)
    }

    logger.info(f"Computed KPIs: {kpis}")
    return kpis

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