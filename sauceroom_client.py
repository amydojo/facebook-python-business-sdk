
import requests
import os
import logging
from typing import Dict, Any, Optional, List
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class SauceRoomClient:
    """Client for integrating with SauceRoom app"""
    
    def __init__(self, base_url: str = "https://sauceroom.replit.app"):
        self.base_url = base_url.rstrip('/')
        self.api_secret = os.getenv('SAUCEROOM_API_KEY', '')
        self.session = requests.Session()
        
        # Set default headers
        if self.api_secret:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_secret}',
                'Content-Type': 'application/json'
            })
    
    def send_social_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Send social media metrics to SauceRoom"""
        try:
            payload = {
                'source': 'campaign_optimizer',
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics
            }
            
            response = self.session.post(
                f"{self.base_url}/api/social/metrics",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Successfully sent metrics to SauceRoom")
                return True
            else:
                logger.warning(f"SauceRoom API returned {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send metrics to SauceRoom: {e}")
            return False
    
    def get_user_engagement(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get user engagement data from SauceRoom"""
        try:
            params = {}
            if user_id:
                params['user_id'] = user_id
            
            response = self.session.get(
                f"{self.base_url}/api/engagement",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"SauceRoom engagement API returned {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get engagement from SauceRoom: {e}")
            return {}
    
    def sync_campaign_data(self, campaign_data: List[Dict[str, Any]]) -> bool:
        """Sync campaign performance data with SauceRoom"""
        try:
            payload = {
                'campaigns': campaign_data,
                'sync_timestamp': datetime.now().isoformat(),
                'source': 'meta_ads'
            }
            
            response = self.session.post(
                f"{self.base_url}/api/campaigns/sync",
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully synced {len(campaign_data)} campaigns to SauceRoom")
                return True
            else:
                logger.warning(f"Campaign sync failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to sync campaigns: {e}")
            return False
    
    def create_audience_segment(self, segment_data: Dict[str, Any]) -> Optional[str]:
        """Create audience segment in SauceRoom based on campaign data"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/audiences/segments",
                json=segment_data,
                timeout=10
            )
            
            if response.status_code == 201:
                result = response.json()
                segment_id = result.get('segment_id')
                logger.info(f"Created audience segment: {segment_id}")
                return segment_id
            else:
                logger.warning(f"Segment creation failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create audience segment: {e}")
            return None
    
    def get_cross_platform_analytics(self) -> Dict[str, Any]:
        """Get combined analytics from SauceRoom"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/analytics/cross-platform",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Analytics API returned {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get cross-platform analytics: {e}")
            return {}

def integrate_with_sauceroom():
    """Main integration function"""
    client = SauceRoomClient()
    
    # Example: Send current Instagram metrics
    try:
        from fetch_organic import fetch_ig_media_insights, get_ig_follower_count
        
        ig_user_id = os.getenv('IG_USER_ID')
        if ig_user_id:
            # Get recent metrics
            follower_count = get_ig_follower_count(ig_user_id)
            
            metrics = {
                'platform': 'instagram',
                'followers': follower_count,
                'timestamp': datetime.now().isoformat()
            }
            
            client.send_social_metrics(metrics)
            
    except Exception as e:
        logger.error(f"Instagram integration failed: {e}")
    
    # Example: Get SauceRoom engagement data
    engagement_data = client.get_user_engagement()
    if engagement_data:
        logger.info(f"Retrieved engagement data: {len(engagement_data.get('users', []))} users")
    
    return client

if __name__ == "__main__":
    # Test the integration
    client = integrate_with_sauceroom()
    
    # Test cross-platform analytics
    analytics = client.get_cross_platform_analytics()
    print(json.dumps(analytics, indent=2))
