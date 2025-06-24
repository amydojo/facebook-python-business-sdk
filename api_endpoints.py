
from flask import Flask, jsonify, request, cors
from flask_cors import CORS
import os
import logging
from datetime import datetime, timedelta
import json
from typing import Dict, Any, Optional

# Import your existing modules
from fetch_organic import fetch_ig_media_insights, get_ig_follower_count, compute_instagram_kpis
from fetch_paid import get_campaign_performance_with_creatives
from api_helpers import get_api_stats

logger = logging.getLogger(__name__)

# Create Flask app for API endpoints
api_app = Flask(__name__)
CORS(api_app)  # Enable CORS for cross-origin requests

# API Authentication (simple token-based)
API_SECRET = os.getenv('API_SECRET', 'your-secret-key-here')

def verify_api_key():
    """Simple API key verification"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return False
    
    token = auth_header.split(' ')[1]
    return token == API_SECRET

@api_app.before_request
def require_api_key():
    """Require API key for all endpoints except health check"""
    if request.endpoint == 'health':
        return
    
    if not verify_api_key():
        return jsonify({'error': 'Invalid or missing API key'}), 401

@api_app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Campaign Optimizer API'
    })

@api_app.route('/api/instagram/insights', methods=['GET'])
def get_instagram_insights():
    """Get Instagram insights for SauceRoom integration"""
    try:
        # Get parameters
        days = request.args.get('days', 7, type=int)
        ig_user_id = os.getenv('IG_USER_ID')
        
        if not ig_user_id:
            return jsonify({'error': 'Instagram User ID not configured'}), 400
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch data
        insights_df = fetch_ig_media_insights(
            ig_user_id,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        follower_count = get_ig_follower_count(ig_user_id)
        
        # Process data for API response
        if insights_df.empty:
            return jsonify({
                'data': [],
                'summary': {
                    'total_posts': 0,
                    'follower_count': follower_count,
                    'date_range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                }
            })
        
        # Convert to API-friendly format
        posts_data = []
        for media_id in insights_df['media_id'].unique():
            post_data = insights_df[insights_df['media_id'] == media_id]
            
            # Create metrics dict
            metrics = {}
            for _, row in post_data.iterrows():
                metrics[row['metric']] = row['value']
            
            post_info = {
                'media_id': media_id,
                'timestamp': post_data.iloc[0]['timestamp'],
                'caption': post_data.iloc[0].get('caption', ''),
                'media_type': post_data.iloc[0].get('media_type', ''),
                'permalink': post_data.iloc[0].get('permalink', ''),
                'metrics': metrics
            }
            
            # Compute KPIs
            try:
                kpis = compute_instagram_kpis(post_data, follower_count)
                post_info['kpis'] = kpis
            except Exception as e:
                logger.warning(f"Could not compute KPIs for {media_id}: {e}")
                post_info['kpis'] = {}
            
            posts_data.append(post_info)
        
        # Summary statistics
        total_reach = sum(post.get('metrics', {}).get('reach', 0) for post in posts_data)
        total_interactions = sum(post.get('metrics', {}).get('total_interactions', 0) for post in posts_data)
        
        summary = {
            'total_posts': len(posts_data),
            'total_reach': total_reach,
            'total_interactions': total_interactions,
            'follower_count': follower_count,
            'avg_engagement_rate': (total_interactions / follower_count * 100) if follower_count else 0,
            'date_range': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        }
        
        return jsonify({
            'data': posts_data,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Instagram insights API error: {e}")
        return jsonify({'error': str(e)}), 500

@api_app.route('/api/campaigns/performance', methods=['GET'])
@app.route('/api/attribution/analysis', methods=['GET'])
@require_api_key
def get_attribution_analysis():
    """Get attribution analysis for SauceRoom integration"""
    try:
        # Get parameters
        attribution_model = request.args.get('model', 'linear')
        date_preset = request.args.get('date_preset', 'last_30d')
        
        # Import attribution functions
        from attribution import (
            build_journeys, first_touch, last_touch, linear_attribution,
            position_based, time_decay, markov_chain_attribution,
            calculate_channel_attribution_summary
        )
        
        # Mock data for demonstration - replace with actual data
        touchpoints = pd.DataFrame({
            'user_id': ['user_1', 'user_1', 'user_2', 'user_2', 'user_3'],
            'channel': ['Instagram_Organic', 'Facebook_Ads', 'Instagram_Organic', 'Facebook_Ads', 'Instagram_Organic'],
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='D'),
            'conversion_id': ['conv_1', 'conv_1', 'conv_2', 'conv_2', 'conv_3'],
            'revenue_amount': [100.0, 100.0, 250.0, 250.0, 75.0]
        })
        
        # Build journeys
        journeys = build_journeys(touchpoints)
        
        if journeys.empty:
            return jsonify({
                'status': 'error',
                'message': 'No journey data available'
            })
        
        # Apply attribution model
        attribution_functions = {
            'first_touch': first_touch,
            'last_touch': last_touch,
            'linear': linear_attribution,
            'position_based': position_based,
            'time_decay': time_decay,
            'markov_chain': markov_chain_attribution
        }
        
        if attribution_model not in attribution_functions:
            return jsonify({
                'status': 'error',
                'message': f'Invalid attribution model: {attribution_model}'
            })
        
        # Run attribution analysis
        attribution_results = attribution_functions[attribution_model](journeys)
        attribution_summary = calculate_channel_attribution_summary(attribution_results)
        
        return jsonify({
            'status': 'success',
            'model': attribution_model,
            'date_preset': date_preset,
            'summary': attribution_summary.to_dict('records'),
            'detailed_results': attribution_results.to_dict('records'),
            'total_attributed_revenue': float(attribution_summary['attributed_revenue'].sum()),
            'channel_count': len(attribution_summary)
        })
        
    except Exception as e:
        logger.error(f"Attribution analysis API error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def get_campaign_performance():
    """Get paid campaign performance for SauceRoom integration"""
    try:
        # Get parameters
        date_preset = request.args.get('date_preset', 'last_7d')
        include_creatives = request.args.get('include_creatives', 'false').lower() == 'true'
        
        # Fetch campaign data
        campaign_data = get_campaign_performance_with_creatives(
            date_preset=date_preset,
            include_creatives=include_creatives
        )
        
        if campaign_data.empty:
            return jsonify({
                'data': [],
                'summary': {
                    'total_campaigns': 0,
                    'total_spend': 0,
                    'total_impressions': 0,
                    'total_clicks': 0,
                    'average_ctr': 0,
                    'date_preset': date_preset
                }
            })
        
        # Convert to API format
        campaigns = campaign_data.to_dict('records')
        
        # Calculate summary
        total_spend = campaign_data['spend'].astype(float).sum()
        total_impressions = campaign_data['impressions'].astype(float).sum()
        total_clicks = campaign_data['clicks'].astype(float).sum()
        avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        
        summary = {
            'total_campaigns': len(campaigns),
            'total_spend': round(total_spend, 2),
            'total_impressions': int(total_impressions),
            'total_clicks': int(total_clicks),
            'average_ctr': round(avg_ctr, 2),
            'date_preset': date_preset
        }
        
        return jsonify({
            'data': campaigns,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Campaign performance API error: {e}")
        return jsonify({'error': str(e)}), 500

@api_app.route('/api/analytics/summary', methods=['GET'])
def get_analytics_summary():
    """Get combined analytics summary for SauceRoom dashboard"""
    try:
        # Get both organic and paid data
        ig_response = get_instagram_insights()
        campaign_response = get_campaign_performance()
        
        ig_data = ig_response[0].get_json() if ig_response[1] == 200 else {}
        campaign_data = campaign_response[0].get_json() if campaign_response[1] == 200 else {}
        
        # Combine into unified summary
        combined_summary = {
            'organic': {
                'posts': ig_data.get('summary', {}).get('total_posts', 0),
                'reach': ig_data.get('summary', {}).get('total_reach', 0),
                'engagement_rate': ig_data.get('summary', {}).get('avg_engagement_rate', 0),
                'followers': ig_data.get('summary', {}).get('follower_count', 0)
            },
            'paid': {
                'campaigns': campaign_data.get('summary', {}).get('total_campaigns', 0),
                'spend': campaign_data.get('summary', {}).get('total_spend', 0),
                'impressions': campaign_data.get('summary', {}).get('total_impressions', 0),
                'ctr': campaign_data.get('summary', {}).get('average_ctr', 0)
            },
            'api_usage': get_api_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(combined_summary)
        
    except Exception as e:
        logger.error(f"Analytics summary API error: {e}")
        return jsonify({'error': str(e)}), 500

@api_app.route('/api/webhook/sauceroom', methods=['POST'])
def sauceroom_webhook():
    """Webhook endpoint for SauceRoom to send data"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Process SauceRoom data
        event_type = data.get('event_type')
        payload = data.get('payload', {})
        
        logger.info(f"Received SauceRoom webhook: {event_type}")
        
        # Handle different event types
        if event_type == 'user_action':
            # Process user action from SauceRoom
            action = payload.get('action')
            user_id = payload.get('user_id')
            
            # You can store this data or trigger campaign optimizations
            logger.info(f"User {user_id} performed action: {action}")
            
        elif event_type == 'engagement':
            # Process engagement data
            engagement_type = payload.get('type')
            metrics = payload.get('metrics', {})
            
            logger.info(f"Engagement event: {engagement_type}, metrics: {metrics}")
        
        # Return success response
        return jsonify({
            'status': 'received',
            'event_type': event_type,
            'processed_at': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Webhook processing error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the API server
    port = int(os.getenv('API_PORT', 5001))
    api_app.run(host='0.0.0.0', port=port, debug=True)
