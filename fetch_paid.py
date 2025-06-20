
"""
Fetch paid advertising insights and leads from Facebook Marketing API.
References:
- Marketing API Insights: https://developers.facebook.com/docs/marketing-api/insights/
- Lead Ads guide: https://developers.facebook.com/docs/marketing-api/guides/lead-ads/
"""
import logging
import pandas as pd
from datetime import datetime, timedelta
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.adobjects.leadgenform import LeadgenForm
from facebook_business.exceptions import FacebookError
from fb_client import fb_client
from config import config

logger = logging.getLogger(__name__)

def fetch_ad_insights(level='campaign', fields=None, date_preset=None, since=None, until=None, 
                     filtering=None, breakdowns=None, time_increment=None):
    """
    Fetch ad insights from Marketing API.
    
    Args:
        level: 'account', 'campaign', 'adset', 'ad'
        fields: list of fields to fetch
        date_preset: 'today', 'yesterday', 'last_7d', 'last_30d', etc.
        since/until: date strings in YYYY-MM-DD format
        filtering: list of filter dictionaries
        breakdowns: list of breakdown fields
        time_increment: 1 (daily), 7 (weekly), monthly
    
    Returns:
        pandas.DataFrame with insights data
        
    Reference: Tested in Graph Explorer: 
    GET /act_{AD_ACCOUNT_ID}/insights?fields=impressions,clicks,spend&date_preset=last_7d
    """
    if not fb_client.is_initialized():
        logger.error("Facebook client not initialized")
        return pd.DataFrame()
    
    # Default fields if none provided
    if fields is None:
        fields = [
            'impressions', 'clicks', 'spend', 'reach', 'frequency',
            'ctr', 'cpc', 'cpm', 'cpp', 'actions', 'action_values',
            'conversions', 'conversion_values', 'cost_per_action_type',
            'cost_per_conversion', 'campaign_name', 'adset_name', 'ad_name'
        ]
    
    # Build parameters
    params = {'level': level, 'fields': fields}
    
    # Date parameters
    if date_preset:
        params['date_preset'] = date_preset
    elif since and until:
        params['time_range'] = {'since': since, 'until': until}
    else:
        # Default to last 7 days
        params['date_preset'] = 'last_7d'
    
    # Optional parameters
    if filtering:
        params['filtering'] = filtering
    if breakdowns:
        params['breakdowns'] = breakdowns
    if time_increment:
        params['time_increment'] = time_increment
    
    try:
        logger.info(f"Fetching {level} insights with params: {params}")
        
        # Get insights from ad account
        insights = fb_client.ad_account.get_insights(params=params)
        
        # Convert to list and handle paging
        insights_data = []
        for insight in insights:
            insights_data.append(dict(insight))
        
        # Handle pagination if there are more results
        while insights.next_page():
            for insight in insights:
                insights_data.append(dict(insight))
        
        if not insights_data:
            logger.warning("No insights data returned")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(insights_data)
        
        # Parse date_start and date_stop to datetime
        if 'date_start' in df.columns:
            df['date_start'] = pd.to_datetime(df['date_start'])
        if 'date_stop' in df.columns:
            df['date_stop'] = pd.to_datetime(df['date_stop'])
        
        # Flatten action columns if present
        df = _flatten_action_columns(df)
        
        logger.info(f"Successfully fetched {len(df)} rows of insights data")
        return df
        
    except FacebookError as e:
        logger.error(f"Error fetching ad insights: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error fetching ad insights: {e}")
        return pd.DataFrame()

def fetch_leads():
    """
    Fetch leads from Lead Generation forms.
    
    Returns:
        pandas.DataFrame with leads data including flattened field_data
        
    Reference: https://developers.facebook.com/docs/marketing-api/guides/lead-ads/
    """
    if not fb_client.is_initialized():
        logger.error("Facebook client not initialized")
        return pd.DataFrame()
    
    try:
        logger.info("Fetching lead generation forms")
        
        # Get leadgen forms
        forms = fb_client.ad_account.get_leadgen_forms(fields=['id', 'name', 'created_time'])
        
        all_leads = []
        
        for form in forms:
            form_id = form['id']
            form_name = form.get('name', 'Unknown Form')
            
            logger.info(f"Fetching leads from form: {form_name} ({form_id})")
            
            # Get leads from this form
            leadgen_form = LeadgenForm(form_id)
            leads = leadgen_form.get_leads(fields=[
                'id', 'field_data', 'created_time', 'ad_id', 'adset_id', 'campaign_id'
            ])
            
            for lead in leads:
                lead_data = dict(lead)
                lead_data['form_id'] = form_id
                lead_data['form_name'] = form_name
                
                # Flatten field_data
                if 'field_data' in lead_data:
                    for field in lead_data['field_data']:
                        field_name = field.get('name', 'unknown_field')
                        field_values = field.get('values', [])
                        # Join multiple values with semicolon
                        lead_data[f"field_{field_name}"] = '; '.join(field_values) if field_values else ''
                    
                    # Remove original field_data
                    del lead_data['field_data']
                
                all_leads.append(lead_data)
        
        if not all_leads:
            logger.warning("No leads found")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_leads)
        
        # Parse created_time
        if 'created_time' in df.columns:
            df['created_time'] = pd.to_datetime(df['created_time'])
        
        logger.info(f"Successfully fetched {len(df)} leads")
        return df
        
    except FacebookError as e:
        logger.error(f"Error fetching leads: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error fetching leads: {e}")
        return pd.DataFrame()

def _flatten_action_columns(df):
    """
    Flatten actions and action_values columns into separate columns.
    
    Args:
        df: DataFrame with potential actions/action_values columns
        
    Returns:
        DataFrame with flattened action columns
    """
    if 'actions' in df.columns:
        df = _flatten_action_column(df, 'actions', 'action')
    
    if 'action_values' in df.columns:
        df = _flatten_action_column(df, 'action_values', 'action_value')
    
    if 'conversions' in df.columns:
        df = _flatten_action_column(df, 'conversions', 'conversion')
    
    if 'conversion_values' in df.columns:
        df = _flatten_action_column(df, 'conversion_values', 'conversion_value')
    
    return df

def _flatten_action_column(df, column_name, prefix):
    """Helper function to flatten action-type columns."""
    if column_name not in df.columns:
        return df
    
    # Create new columns for each action type
    action_types = set()
    for actions in df[column_name].dropna():
        if isinstance(actions, list):
            for action in actions:
                if isinstance(action, dict) and 'action_type' in action:
                    action_types.add(action['action_type'])
    
    # Add columns for each action type
    for action_type in action_types:
        column = f"{prefix}_{action_type}"
        df[column] = df[column_name].apply(
            lambda x: _extract_action_value(x, action_type) if x else 0
        )
    
    return df

def _extract_action_value(actions, target_action_type):
    """Extract value for specific action type from actions list."""
    if not isinstance(actions, list):
        return 0
    
    for action in actions:
        if isinstance(action, dict) and action.get('action_type') == target_action_type:
            return float(action.get('value', 0))
    
    return 0

def get_campaign_performance_summary(days=7):
    """
    Get a summary of campaign performance for the specified number of days.
    
    Args:
        days: Number of days to look back
        
    Returns:
        pandas.DataFrame with campaign performance summary
    """
    # Calculate date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    # Fetch campaign insights
    df = fetch_ad_insights(
        level='campaign',
        since=start_date.strftime('%Y-%m-%d'),
        until=end_date.strftime('%Y-%m-%d'),
        fields=['campaign_name', 'impressions', 'clicks', 'spend', 'ctr', 'cpc', 'cpm']
    )
    
    if df.empty:
        return df
    
    # Calculate additional metrics
    df['cost_per_click'] = pd.to_numeric(df.get('spend', 0)) / pd.to_numeric(df.get('clicks', 1))
    df['cost_per_mille'] = pd.to_numeric(df.get('spend', 0)) / pd.to_numeric(df.get('impressions', 1)) * 1000
    
    return df
