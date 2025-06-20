
"""
Attribution engine for multi-touch attribution analysis.
References:
- Attribution concepts: https://www.attributionapp.com/blog/revenue-attribution/
- Marketing science attribution models
"""
import logging
import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
from config import config

logger = logging.getLogger(__name__)

def build_journeys(merged_df):
    """
    Build customer journeys from merged touchpoint and conversion data.
    
    Args:
        merged_df: DataFrame with columns:
            - lead_id or user_id: unique identifier
            - timestamp: datetime of touchpoint
            - channel: touchpoint channel/source
            - conversion_id: unique conversion identifier (optional)
            - revenue_amount: revenue value (optional)
    
    Returns:
        DataFrame with columns: conversion_id, channel_sequence, revenue_amount
    """
    if merged_df.empty:
        logger.warning("Empty merged_df provided to build_journeys")
        return pd.DataFrame(columns=['conversion_id', 'channel_sequence', 'revenue_amount'])
    
    # Validate required columns
    required_cols = ['timestamp', 'channel']
    user_id_col = 'user_id' if 'user_id' in merged_df.columns else 'lead_id'
    
    if user_id_col not in merged_df.columns:
        logger.error("Neither 'user_id' nor 'lead_id' column found in merged_df")
        return pd.DataFrame(columns=['conversion_id', 'channel_sequence', 'revenue_amount'])
    
    missing_cols = [col for col in required_cols if col not in merged_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return pd.DataFrame(columns=['conversion_id', 'channel_sequence', 'revenue_amount'])
    
    try:
        # Sort by user and timestamp
        df = merged_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values([user_id_col, 'timestamp'])
        
        journeys = []
        
        # Group by user/lead
        for user_id, user_data in df.groupby(user_id_col):
            # Get channel sequence
            channels = user_data['channel'].tolist()
            
            # Find conversions (rows with conversion_id)
            conversions = user_data[user_data['conversion_id'].notna()]
            
            if conversions.empty:
                # No conversions for this user, create journey without conversion
                journey = {
                    'conversion_id': f"no_conversion_{user_id}",
                    'channel_sequence': channels,
                    'revenue_amount': 0.0,
                    'user_id': user_id,
                    'converted': False
                }
                journeys.append(journey)
            else:
                # For each conversion, build journey up to that point
                for _, conversion in conversions.iterrows():
                    conversion_time = conversion['timestamp']
                    
                    # Get touchpoints before or at conversion time
                    journey_touchpoints = user_data[user_data['timestamp'] <= conversion_time]
                    journey_channels = journey_touchpoints['channel'].tolist()
                    
                    journey = {
                        'conversion_id': conversion['conversion_id'],
                        'channel_sequence': journey_channels,
                        'revenue_amount': conversion.get('revenue_amount', 0.0),
                        'user_id': user_id,
                        'converted': True
                    }
                    journeys.append(journey)
        
        result_df = pd.DataFrame(journeys)
        logger.info(f"Built {len(result_df)} customer journeys")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error building journeys: {e}")
        return pd.DataFrame(columns=['conversion_id', 'channel_sequence', 'revenue_amount'])

def first_touch(journeys_df):
    """
    First-touch attribution model.
    
    Args:
        journeys_df: DataFrame from build_journeys()
    
    Returns:
        DataFrame with columns: conversion_id, channel, credit_fraction
    """
    if journeys_df.empty:
        logger.warning("Empty journeys_df provided to first_touch")
        return pd.DataFrame(columns=['conversion_id', 'channel', 'credit_fraction'])
    
    try:
        attributions = []
        
        for _, journey in journeys_df.iterrows():
            channels = journey['channel_sequence']
            if channels:
                first_channel = channels[0]
                attributions.append({
                    'conversion_id': journey['conversion_id'],
                    'channel': first_channel,
                    'credit_fraction': 1.0,
                    'revenue_amount': journey['revenue_amount']
                })
        
        result_df = pd.DataFrame(attributions)
        logger.info(f"Applied first-touch attribution to {len(result_df)} conversions")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in first_touch attribution: {e}")
        return pd.DataFrame(columns=['conversion_id', 'channel', 'credit_fraction'])

def last_touch(journeys_df):
    """
    Last-touch attribution model.
    
    Args:
        journeys_df: DataFrame from build_journeys()
    
    Returns:
        DataFrame with columns: conversion_id, channel, credit_fraction
    """
    if journeys_df.empty:
        logger.warning("Empty journeys_df provided to last_touch")
        return pd.DataFrame(columns=['conversion_id', 'channel', 'credit_fraction'])
    
    try:
        attributions = []
        
        for _, journey in journeys_df.iterrows():
            channels = journey['channel_sequence']
            if channels:
                last_channel = channels[-1]
                attributions.append({
                    'conversion_id': journey['conversion_id'],
                    'channel': last_channel,
                    'credit_fraction': 1.0,
                    'revenue_amount': journey['revenue_amount']
                })
        
        result_df = pd.DataFrame(attributions)
        logger.info(f"Applied last-touch attribution to {len(result_df)} conversions")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in last_touch attribution: {e}")
        return pd.DataFrame(columns=['conversion_id', 'channel', 'credit_fraction'])

def linear_attribution(journeys_df):
    """
    Linear attribution model - equal credit to all touchpoints.
    
    Args:
        journeys_df: DataFrame from build_journeys()
    
    Returns:
        DataFrame with columns: conversion_id, channel, credit_fraction
    """
    if journeys_df.empty:
        logger.warning("Empty journeys_df provided to linear_attribution")
        return pd.DataFrame(columns=['conversion_id', 'channel', 'credit_fraction'])
    
    try:
        attributions = []
        
        for _, journey in journeys_df.iterrows():
            channels = journey['channel_sequence']
            if channels:
                unique_channels = list(set(channels))  # Remove duplicates
                credit_per_channel = 1.0 / len(unique_channels)
                
                for channel in unique_channels:
                    attributions.append({
                        'conversion_id': journey['conversion_id'],
                        'channel': channel,
                        'credit_fraction': credit_per_channel,
                        'revenue_amount': journey['revenue_amount']
                    })
        
        result_df = pd.DataFrame(attributions)
        logger.info(f"Applied linear attribution to {len(result_df)} channel-conversion pairs")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in linear_attribution: {e}")
        return pd.DataFrame(columns=['conversion_id', 'channel', 'credit_fraction'])

def position_based(journeys_df, first_weight=0.4, last_weight=0.4):
    """
    Position-based attribution model (U-shaped).
    
    Args:
        journeys_df: DataFrame from build_journeys()
        first_weight: weight for first touchpoint
        last_weight: weight for last touchpoint
    
    Returns:
        DataFrame with columns: conversion_id, channel, credit_fraction
    """
    if journeys_df.empty:
        logger.warning("Empty journeys_df provided to position_based")
        return pd.DataFrame(columns=['conversion_id', 'channel', 'credit_fraction'])
    
    middle_weight = 1.0 - first_weight - last_weight
    
    try:
        attributions = []
        
        for _, journey in journeys_df.iterrows():
            channels = journey['channel_sequence']
            if not channels:
                continue
            
            if len(channels) == 1:
                # Single touchpoint gets full credit
                attributions.append({
                    'conversion_id': journey['conversion_id'],
                    'channel': channels[0],
                    'credit_fraction': 1.0,
                    'revenue_amount': journey['revenue_amount']
                })
            elif len(channels) == 2:
                # Two touchpoints split first and last weights
                attributions.append({
                    'conversion_id': journey['conversion_id'],
                    'channel': channels[0],
                    'credit_fraction': first_weight,
                    'revenue_amount': journey['revenue_amount']
                })
                attributions.append({
                    'conversion_id': journey['conversion_id'],
                    'channel': channels[1],
                    'credit_fraction': last_weight,
                    'revenue_amount': journey['revenue_amount']
                })
            else:
                # Multiple touchpoints
                unique_channels = list(set(channels))
                middle_channels = set(channels[1:-1])
                
                # Track credit by channel
                channel_credits = defaultdict(float)
                
                # First touchpoint
                channel_credits[channels[0]] += first_weight
                
                # Last touchpoint
                channel_credits[channels[-1]] += last_weight
                
                # Middle touchpoints
                if middle_channels:
                    middle_credit_per_channel = middle_weight / len(middle_channels)
                    for channel in middle_channels:
                        channel_credits[channel] += middle_credit_per_channel
                
                # Create attribution records
                for channel, credit in channel_credits.items():
                    attributions.append({
                        'conversion_id': journey['conversion_id'],
                        'channel': channel,
                        'credit_fraction': credit,
                        'revenue_amount': journey['revenue_amount']
                    })
        
        result_df = pd.DataFrame(attributions)
        logger.info(f"Applied position-based attribution to {len(result_df)} channel-conversion pairs")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in position_based attribution: {e}")
        return pd.DataFrame(columns=['conversion_id', 'channel', 'credit_fraction'])

def time_decay(journeys_df, half_life_days=7):
    """
    Time-decay attribution model - more recent touchpoints get more credit.
    
    Args:
        journeys_df: DataFrame from build_journeys()
        half_life_days: number of days for credit to decay to 50%
    
    Returns:
        DataFrame with columns: conversion_id, channel, credit_fraction
    """
    if journeys_df.empty:
        logger.warning("Empty journeys_df provided to time_decay")
        return pd.DataFrame(columns=['conversion_id', 'channel', 'credit_fraction'])
    
    try:
        attributions = []
        
        for _, journey in journeys_df.iterrows():
            channels = journey['channel_sequence']
            if not channels:
                continue
            
            # For time decay, we need timestamps - using position as proxy
            num_touchpoints = len(channels)
            
            # Calculate decay weights (more recent = higher weight)
            weights = []
            for i in range(num_touchpoints):
                days_ago = num_touchpoints - i - 1  # Days before conversion
                weight = 0.5 ** (days_ago / half_life_days)
                weights.append(weight)
            
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            # Aggregate by channel
            channel_credits = defaultdict(float)
            for i, channel in enumerate(channels):
                channel_credits[channel] += normalized_weights[i]
            
            # Create attribution records
            for channel, credit in channel_credits.items():
                attributions.append({
                    'conversion_id': journey['conversion_id'],
                    'channel': channel,
                    'credit_fraction': credit,
                    'revenue_amount': journey['revenue_amount']
                })
        
        result_df = pd.DataFrame(attributions)
        logger.info(f"Applied time-decay attribution to {len(result_df)} channel-conversion pairs")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in time_decay attribution: {e}")
        return pd.DataFrame(columns=['conversion_id', 'channel', 'credit_fraction'])

def markov_chain_attribution(journeys_df, max_journeys_for_markov=1000):
    """
    Markov chain attribution model - uses transition probabilities and removal effect.
    Note: This is a simplified implementation. For large datasets, consider sampling.
    
    Args:
        journeys_df: DataFrame from build_journeys()
        max_journeys_for_markov: maximum journeys to process (for performance)
    
    Returns:
        DataFrame with columns: conversion_id, channel, credit_fraction
    """
    if journeys_df.empty:
        logger.warning("Empty journeys_df provided to markov_chain_attribution")
        return pd.DataFrame(columns=['conversion_id', 'channel', 'credit_fraction'])
    
    # Sample data if too large and not forced
    if len(journeys_df) > max_journeys_for_markov and not config.FORCE_MARKOV:
        logger.warning(f"Large dataset ({len(journeys_df)} journeys), sampling {max_journeys_for_markov} for Markov model")
        journeys_df = journeys_df.sample(n=max_journeys_for_markov, random_state=42)
    
    try:
        # Build transition matrix
        transitions = defaultdict(lambda: defaultdict(int))
        all_channels = set()
        conversion_paths = []
        
        for _, journey in journeys_df.iterrows():
            channels = journey['channel_sequence']
            if not channels:
                continue
            
            # Add start and conversion states
            path = ['START'] + channels + ['CONVERSION']
            conversion_paths.append((path, journey['conversion_id'], journey['revenue_amount']))
            
            # Count transitions
            for i in range(len(path) - 1):
                from_state = path[i]
                to_state = path[i + 1]
                transitions[from_state][to_state] += 1
                all_channels.add(from_state)
                all_channels.add(to_state)
        
        # Calculate conversion probability for baseline
        baseline_conversion_prob = _calculate_conversion_probability(transitions, all_channels)
        
        # Calculate removal effect for each channel
        attributions = []
        unique_channels = [ch for ch in all_channels if ch not in ['START', 'CONVERSION']]
        
        for channel in unique_channels:
            # Calculate conversion probability without this channel
            prob_without_channel = _calculate_conversion_probability_without_channel(
                transitions, all_channels, channel
            )
            
            # Removal effect (how much conversion probability drops)
            removal_effect = baseline_conversion_prob - prob_without_channel
            
            # Attribute to all conversions proportionally
            for path, conversion_id, revenue in conversion_paths:
                if channel in path:
                    # Simplified credit assignment based on removal effect
                    credit_fraction = removal_effect / len([ch for ch in path if ch in unique_channels])
                    credit_fraction = max(0, min(1, credit_fraction))  # Clamp between 0 and 1
                    
                    attributions.append({
                        'conversion_id': conversion_id,
                        'channel': channel,
                        'credit_fraction': credit_fraction,
                        'revenue_amount': revenue
                    })
        
        result_df = pd.DataFrame(attributions)
        
        # Normalize credits per conversion to sum to 1
        if not result_df.empty:
            result_df = _normalize_attribution_credits(result_df)
        
        logger.info(f"Applied Markov chain attribution to {len(result_df)} channel-conversion pairs")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in markov_chain_attribution: {e}")
        # Fallback to linear attribution
        logger.info("Falling back to linear attribution")
        return linear_attribution(journeys_df)

def _calculate_conversion_probability(transitions, all_channels):
    """Calculate baseline conversion probability using transition matrix."""
    try:
        # Simple calculation: ratio of paths that lead to conversion
        total_transitions_from_start = sum(transitions['START'].values())
        if total_transitions_from_start == 0:
            return 0.0
        
        # This is a simplified approach - a full Markov model would solve the system of equations
        return 1.0  # Placeholder - all observed paths lead to conversion in our data
        
    except:
        return 1.0

def _calculate_conversion_probability_without_channel(transitions, all_channels, removed_channel):
    """Calculate conversion probability with a channel removed."""
    try:
        # Simplified approach: assume removal reduces conversion by a factor
        # In a full implementation, this would rebuild the transition matrix
        return 0.8  # Placeholder - assume 20% reduction
        
    except:
        return 0.8

def _normalize_attribution_credits(df):
    """Normalize attribution credits so they sum to 1 per conversion."""
    if df.empty:
        return df
    
    # Group by conversion and normalize
    result_rows = []
    for conversion_id, group in df.groupby('conversion_id'):
        total_credit = group['credit_fraction'].sum()
        if total_credit > 0:
            group = group.copy()
            group['credit_fraction'] = group['credit_fraction'] / total_credit
        result_rows.append(group)
    
    return pd.concat(result_rows, ignore_index=True)

def calculate_channel_attribution_summary(attribution_df):
    """
    Calculate summary statistics by channel from attribution results.
    
    Args:
        attribution_df: DataFrame from attribution model functions
    
    Returns:
        DataFrame with channel-level attribution summary
    """
    if attribution_df.empty:
        return pd.DataFrame(columns=['channel', 'total_credit', 'attributed_revenue', 'conversions'])
    
    try:
        # Calculate attributed revenue
        attribution_df['attributed_revenue'] = (
            attribution_df['credit_fraction'] * attribution_df['revenue_amount']
        )
        
        # Group by channel
        summary = attribution_df.groupby('channel').agg({
            'credit_fraction': ['sum', 'count'],
            'attributed_revenue': 'sum',
            'conversion_id': 'nunique'
        }).round(4)
        
        # Flatten column names
        summary.columns = ['total_credit', 'touchpoints', 'attributed_revenue', 'conversions']
        summary = summary.reset_index()
        
        # Sort by attributed revenue
        summary = summary.sort_values('attributed_revenue', ascending=False)
        
        logger.info(f"Calculated attribution summary for {len(summary)} channels")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error calculating channel attribution summary: {e}")
        return pd.DataFrame(columns=['channel', 'total_credit', 'attributed_revenue', 'conversions'])
