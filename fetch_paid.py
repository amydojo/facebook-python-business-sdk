
"""
Facebook Ads API integration for fetching paid campaign performance data.
Reference: https://developers.facebook.com/docs/marketing-api/insights/
"""
import logging
from datetime import datetime, timedelta
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.adobjects.adsinsights import AdsInsights
from facebook_business.exceptions import FacebookError
from fb_client import fb_client
from data_store import data_store

logger = logging.getLogger(__name__)

def get_campaign_performance(date_preset='last_7d', level='campaign', fields=None):
    """
    Fetch campaign performance data from Facebook Ads API.
    
    Args:
        date_preset: Date range preset (last_7d, last_30d, etc.)
        level: Reporting level (campaign, adset, ad)
        fields: List of fields to retrieve
    
    Returns:
        list: Campaign performance data or empty list if error
    """
    if not fb_client.is_initialized():
        logger.error("Facebook client not initialized - cannot fetch campaign data")
        return []
    
    if fields is None:
        fields = [
            'campaign_id',
            'campaign_name', 
            'impressions',
            'clicks',
            'spend',
            'reach',
            'frequency',
            'ctr',
            'cpc',
            'cpm',
            'cpp',
            'date_start',
            'date_stop'
        ]
    
    try:
        ad_account = fb_client.get_ad_account()
        if not ad_account:
            logger.error("No ad account available")
            return []
        
        logger.info(f"Fetching {level} insights for date preset: {date_preset}")
        
        # Get insights from Facebook API
        insights = ad_account.get_insights(
            fields=fields,
            params={
                'date_preset': date_preset,
                'level': level,
                'limit': 100
            }
        )
        
        performance_data = []
        for insight in insights:
            data = dict(insight)
            
            # Store in database
            data_store.store_performance_data(
                entity_type=level,
                entity_id=data.get('campaign_id') or data.get('adset_id') or data.get('ad_id'),
                entity_name=data.get('campaign_name') or data.get('adset_name') or data.get('ad_name'),
                data=data,
                source='facebook_ads'
            )
            
            performance_data.append(data)
        
        logger.info(f"Successfully fetched {len(performance_data)} {level} records")
        return performance_data
        
    except FacebookError as e:
        logger.error(f"Facebook API error: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching campaign performance: {e}")
        return []

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
        performance_data = get_campaign_performance(date_preset=date_preset, level='campaign')
        
        if not performance_data:
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
            performance_data = [
                data for data in performance_data 
                if data.get('campaign_id') in campaign_ids
            ]
        
        # Calculate summary metrics
        total_spend = sum(float(data.get('spend', 0)) for data in performance_data)
        total_impressions = sum(int(data.get('impressions', 0)) for data in performance_data)
        total_clicks = sum(int(data.get('clicks', 0)) for data in performance_data)
        
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
        
        logger.info(f"Generated performance summary: {summary}")
        return summary
        
    except Exception as e:
        logger.error(f"Error generating campaign performance summary: {e}")
        return {}

def get_ad_performance(campaign_id=None, date_preset='last_7d'):
    """
    Get ad-level performance data.
    
    Args:
        campaign_id: Specific campaign ID to filter by
        date_preset: Date range preset
    
    Returns:
        list: Ad performance data
    """
    if not fb_client.is_initialized():
        logger.error("Facebook client not initialized")
        return []
    
    try:
        params = {
            'date_preset': date_preset,
            'level': 'ad',
            'limit': 100
        }
        
        if campaign_id:
            params['filtering'] = [{'field': 'campaign.id', 'operator': 'IN', 'value': [campaign_id]}]
        
        ad_account = fb_client.get_ad_account()
        insights = ad_account.get_insights(
            fields=[
                'ad_id',
                'ad_name',
                'campaign_id',
                'campaign_name',
                'adset_id',
                'adset_name',
                'impressions',
                'clicks',
                'spend',
                'ctr',
                'cpc',
                'cpm'
            ],
            params=params
        )
        
        ad_data = []
        for insight in insights:
            data = dict(insight)
            
            # Store in database
            data_store.store_performance_data(
                entity_type='ad',
                entity_id=data.get('ad_id'),
                entity_name=data.get('ad_name'),
                data=data,
                source='facebook_ads'
            )
            
            ad_data.append(data)
        
        logger.info(f"Fetched {len(ad_data)} ad performance records")
        return ad_data
        
    except Exception as e:
        logger.error(f"Error fetching ad performance: {e}")
        return []

def get_real_time_insights(campaign_ids=None):
    """
    Get real-time campaign insights (last hour data).
    
    Args:
        campaign_ids: List of campaign IDs to check
    
    Returns:
        list: Real-time insights data
    """
    if not fb_client.is_initialized():
        logger.error("Facebook client not initialized")
        return []
    
    try:
        ad_account = fb_client.get_ad_account()
        
        params = {
            'date_preset': 'today',
            'level': 'campaign',
            'time_increment': 1  # Hourly breakdown
        }
        
        if campaign_ids:
            params['filtering'] = [{'field': 'campaign.id', 'operator': 'IN', 'value': campaign_ids}]
        
        insights = ad_account.get_insights(
            fields=[
                'campaign_id',
                'campaign_name',
                'impressions',
                'clicks',
                'spend',
                'date_start',
                'date_stop'
            ],
            params=params
        )
        
        real_time_data = []
        for insight in insights:
            data = dict(insight)
            real_time_data.append(data)
        
        logger.info(f"Fetched {len(real_time_data)} real-time insight records")
        return real_time_data
        
    except Exception as e:
        logger.error(f"Error fetching real-time insights: {e}")
        return []
