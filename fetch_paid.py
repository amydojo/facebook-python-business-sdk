
"""
Optimized Facebook Ads API integration with rate limiting and performance improvements.
Official docs: https://developers.facebook.com/docs/marketing-api/insights/
"""
import os
import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from config import config

# Import optimized API helpers
from api_helpers import safe_api_call, batch_facebook_requests, get_api_stats

# Import fb_client for API access
from fb_client import fb_client

# Facebook Business SDK imports
try:
    from facebook_business.adobjects.campaign import Campaign
    from facebook_business.adobjects.ad import Ad
    from facebook_business.adobjects.adcreative import AdCreative
    from facebook_business.adobjects.adaccount import AdAccount
    CREATIVE_SDK_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Creative SDK imports not available: {e}")
    CREATIVE_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)

def get_account_with_retries() -> Optional[AdAccount]:
    """Get Facebook Ad Account with safe API call."""
    if not hasattr(fb_client, "account") or fb_client.account is None:
        logger.error("❌ Facebook client not initialized")
        return None
    return fb_client.account

def fetch_campaign_insights_optimized(
    level: str = "campaign",
    fields: List[str] = None,
    date_preset: str = None,
    since: str = None,
    until: str = None,
    filtering: List[Dict] = None,
    breakdowns: List[str] = None,
    use_cache: bool = True,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Optimized insights fetcher using expanded fields and safe API calls.
    
    Args:
        level: 'campaign', 'adset', or 'ad'
        fields: List of metrics and expanded fields
        date_preset: 'yesterday', 'last_7d', 'last_30d', etc.
        since/until: Custom date range (YYYY-MM-DD format)
        filtering: List of filters
        breakdowns: List of breakdowns
        use_cache: Whether to use caching
        force_refresh: Force refresh cache
    
    Returns:
        pandas.DataFrame with insights data
    """
    account = get_account_with_retries()
    if account is None:
        return pd.DataFrame()

    # Enhanced field lists with creative expansion
    if fields is None:
        base_fields = ["impressions", "clicks", "spend", "reach", "frequency", "ctr", "cpc", "cpm"]
        
        if level == "campaign":
            fields = [
                "campaign_id", "campaign_name", "objective", "status",
                *base_fields,
                "date_start", "date_stop"
            ]
        elif level == "adset":
            fields = [
                "adset_id", "adset_name", "campaign_id", "campaign_name",
                "optimization_goal", "billing_event", "bid_amount",
                *base_fields,
                "date_start", "date_stop"
            ]
        else:  # ad level
            fields = [
                "ad_id", "ad_name", "adset_id", "adset_name", "campaign_id", "campaign_name",
                *base_fields,
                "date_start", "date_stop"
                # Note: creative fields cannot be fetched in insights call
                # Will fetch separately using ad creative endpoint
            ]

    # Build params
    params = {"level": level}

    if date_preset:
        params["date_preset"] = date_preset
    elif since and until:
        params["time_range"] = {"since": since, "until": until}
    else:
        params["date_preset"] = "yesterday"

    if filtering:
        params["filtering"] = filtering
    if breakdowns:
        params["breakdowns"] = breakdowns

    # Create cache-friendly endpoint identifier
    endpoint = f"insights_{level}_{date_preset or f'{since}_to_{until}'}"
    
    def api_call():
        return account.get_insights(fields=fields, params=params)

    # Use safe API call with caching
    insights_data = safe_api_call(
        api_call,
        endpoint,
        params,
        use_cache=use_cache,
        cache_ttl_hours=2,  # Cache for 2 hours
        force_refresh=force_refresh
    )

    if not insights_data:
        logger.warning("⚠️ No insights data returned")
        return pd.DataFrame()

    # Process paginated results
    all_data = []
    page_count = 0
    
    try:
        # Handle both cached data (list) and live API response
        if isinstance(insights_data, list):
            all_data = insights_data
        else:
            while insights_data:
                page_count += 1
                logger.info(f"📄 Processing insights page {page_count}")
                
                for entry in insights_data:
                    data = entry.export_all_data() if hasattr(entry, 'export_all_data') else entry
                    all_data.append(data)
                
                try:
                    # Rate-limited pagination
                    insights_data = safe_api_call(
                        lambda: insights_data.next_page(),
                        f"{endpoint}_page_{page_count}",
                        {},
                        use_cache=False  # Don't cache pagination
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

def enrich_with_creative_data_batch(df_ads: pd.DataFrame) -> pd.DataFrame:
    """
    Batch enrich ad data with creative information.
    
    Args:
        df_ads: DataFrame with ad performance data
    
    Returns:
        DataFrame enriched with creative data
    """
    if df_ads.empty or not CREATIVE_SDK_AVAILABLE:
        return df_ads

    # If creative data is already present, return as-is
    if any('creative_' in col for col in df_ads.columns):
        logger.info("Creative data already present in response")
        return df_ads

    # Get unique ad IDs, limit for safety
    unique_ad_ids = df_ads['ad_id'].unique()[:50]  
    logger.info(f"Fetching creative data for {len(unique_ad_ids)} ads")
    
    def batch_get_creatives():
        creative_data = {}
        for ad_id in unique_ad_ids:
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
                    
                    # Now fetch the actual creative details
                    def get_creative_details():
                        creative = AdCreative(creative_id)
                        return creative.api_get(fields=[
                            'id', 'name', 'body', 'title', 'image_url', 
                            'thumbnail_url', 'object_url'
                        ])
                    
                    creative_details = safe_api_call(
                        get_creative_details,
                        f"creative_{creative_id}",
                        {},
                        cache_ttl_hours=24
                    )
                    
                    if creative_details:
                        creative_data[ad_id] = creative_details.export_all_data() if hasattr(creative_details, 'export_all_data') else creative_details
                    
            except Exception as e:
                logger.warning(f"Failed to fetch creative for ad {ad_id}: {e}")
                continue
        
        return creative_data

    creative_data = batch_get_creatives()
    
    # Merge creative data with performance data
    if creative_data:
        logger.info(f"Enriching {len(df_ads)} rows with creative data from {len(creative_data)} ads")
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
        
        return pd.DataFrame(creative_rows)
    
    return df_ads

def get_campaign_performance_optimized(
    date_preset: str = "last_7d",
    since: str = None,
    until: str = None,
    extra_fields: List[str] = None,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Optimized campaign performance fetcher.
    
    Args:
        date_preset: 'yesterday', 'last_7d', 'last_30d', etc.
        since/until: Custom date range
        extra_fields: Additional fields to include
        force_refresh: Force refresh cache
    
    Returns:
        DataFrame with campaign performance data
    """
    logger.info(f"📊 Fetching optimized campaign performance for {date_preset or f'{since} to {until}'}")

    base_fields = [
        "campaign_id", "campaign_name", "objective", "status",
        "impressions", "clicks", "spend", "reach", "frequency", 
        "ctr", "cpc", "cpm", "date_start", "date_stop"
    ]

    if extra_fields:
        base_fields.extend([f for f in extra_fields if f not in base_fields])

    return fetch_campaign_insights_optimized(
        level="campaign",
        fields=base_fields,
        date_preset=date_preset,
        since=since,
        until=until,
        force_refresh=force_refresh
    )

def get_ad_performance_optimized(
    campaign_ids: List[str] = None,
    date_preset: str = "last_7d",
    include_creatives: bool = True,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Optimized ad-level performance with creative data.
    
    Args:
        campaign_ids: Filter by specific campaign IDs
        date_preset: Date range preset
        include_creatives: Include creative preview data
        force_refresh: Force refresh cache
    
    Returns:
        DataFrame with ad performance and creative data
    """
    filtering = None
    if campaign_ids:
        filtering = [{'field': 'campaign.id', 'operator': 'IN', 'value': campaign_ids}]

    # Enhanced fields with creative expansion
    fields = [
        "ad_id", "ad_name", "adset_id", "adset_name", "campaign_id", "campaign_name",
        "impressions", "clicks", "spend", "reach", "ctr", "cpc", "cpm",
        "date_start", "date_stop"
    ]
    
    # Note: Cannot include creative fields in insights call
    # Will enrich with creative data after getting insights
    
    df_ads = fetch_campaign_insights_optimized(
        level="ad",
        fields=fields,
        date_preset=date_preset,
        filtering=filtering,
        force_refresh=force_refresh
    )

    # Enrich with creative data if requested
    if include_creatives and not df_ads.empty:
        df_ads = enrich_with_creative_data_batch(df_ads)

    return df_ads

def get_campaign_performance_with_creatives(
    date_preset: str = "last_7d",
    include_creatives: bool = True,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Main function for campaign performance with creative data.
    
    Args:
        date_preset: Date range preset
        include_creatives: Whether to fetch creative preview data
        force_refresh: Force refresh cache
    
    Returns:
        DataFrame with campaign performance and creative data
    """
    if include_creatives:
        # Get ad-level data with creatives (more comprehensive)
        return get_ad_performance_optimized(
            date_preset=date_preset,
            include_creatives=True,
            force_refresh=force_refresh
        )
    else:
        # Get campaign-level data only
        return get_campaign_performance_optimized(
            date_preset=date_preset,
            force_refresh=force_refresh
        )

def get_paid_insights(
    date_preset: str = "last_7d",
    since: str = None,
    until: str = None,
    include_creatives: bool = True,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Main function to get paid campaign insights with optimizations.
    
    Args:
        date_preset: Date range preset
        since/until: Custom date range
        include_creatives: Include creative preview data
        force_refresh: Force refresh cache
    
    Returns:
        DataFrame with paid campaign performance and creative data
    """
    try:
        logger.info(f"📊 Fetching optimized paid insights for {date_preset or f'{since} to {until}'}")

        if since and until:
            # Custom date range - campaign level only for performance
            return fetch_campaign_insights_optimized(
                level="campaign",
                date_preset=None,
                since=since,
                until=until,
                force_refresh=force_refresh
            )
        else:
            # Use optimized function with creative data
            return get_campaign_performance_with_creatives(
                date_preset=date_preset,
                include_creatives=include_creatives,
                force_refresh=force_refresh
            )

    except Exception as e:
        logger.error(f"❌ Error fetching optimized paid insights: {e}", exc_info=True)
        return pd.DataFrame()

# Backward compatibility aliases
def get_campaign_insights(*args, **kwargs):
    """Alias for get_paid_insights"""
    return get_paid_insights(*args, **kwargs)

def get_campaign_performance_summary(
    date_preset: str = "last_7d",
    campaign_ids: List[str] = None,
    force_refresh: bool = False
) -> Dict:
    """
    Optimized campaign performance summary.
    
    Args:
        date_preset: Date range preset
        campaign_ids: Filter by specific campaign IDs
        force_refresh: Force refresh cache
    
    Returns:
        Dict with summary statistics
    """
    try:
        filtering = None
        if campaign_ids:
            filtering = [{'field': 'campaign.id', 'operator': 'IN', 'value': campaign_ids}]

        performance_data = fetch_campaign_insights_optimized(
            level="campaign",
            date_preset=date_preset,
            filtering=filtering,
            force_refresh=force_refresh
        )

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

        logger.info(f"📈 Generated optimized performance summary: {summary}")
        return summary

    except Exception as e:
        logger.error(f"❌ Error generating optimized performance summary: {e}", exc_info=True)
        return {'api_stats': get_api_stats()}

if __name__ == "__main__":
    # Test optimized paid insights
    logger.info("🧪 Testing optimized paid insights...")
    try:
        df_paid = get_paid_insights(date_preset="last_7d", force_refresh=True)
        print(f"Optimized paid insights: {len(df_paid)} records")
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
