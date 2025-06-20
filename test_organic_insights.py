
#!/usr/bin/env python3
"""
Test script for organic insights functionality.
Run this to test Facebook Page and Instagram insights fetching.

Usage: python test_organic_insights.py
"""
import os
import logging
from datetime import date, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_organic_insights():
    """Test organic insights fetching functionality."""
    
    logger.info("🧪 Testing organic insights functionality")
    
    try:
        from fetch_organic import (
            validate_organic_environment,
            fetch_latest_page_insights,
            fetch_latest_ig_media_insights,
            get_organic_insights,
            fetch_page_insights_metadata
        )
        
        # Validate environment
        logger.info("Validating environment...")
        validation = validate_organic_environment()
        logger.info(f"Environment validation: {validation}")
        
        if not validation['page_insights_enabled']:
            logger.warning("⚠️ Page insights disabled - set PAGE_ACCESS_TOKEN and PAGE_ID")
        
        if not validation['instagram_insights_enabled']:
            logger.warning("⚠️ Instagram insights disabled - set PAGE_ACCESS_TOKEN and IG_USER_ID")
        
        # Test metadata fetching
        if validation['page_insights_enabled']:
            logger.info("Fetching page insights metadata...")
            metadata = fetch_page_insights_metadata()
            if metadata:
                logger.info("✅ Page insights metadata fetched successfully")
            else:
                logger.warning("⚠️ Could not fetch page insights metadata")
        
        # Test latest page insights
        if validation['page_insights_enabled']:
            logger.info("Testing latest page insights...")
            page_metrics = ['page_impressions_organic', 'page_engaged_users', 'page_reach']
            df_page = fetch_latest_page_insights(page_metrics)
            
            if not df_page.empty:
                logger.info(f"✅ Latest Page insights: {len(df_page)} records")
                print("Latest Page insights sample:")
                print(df_page.head())
            else:
                logger.warning("⚠️ No latest page insights data")
        
        # Test latest Instagram insights
        if validation['instagram_insights_enabled']:
            logger.info("Testing latest Instagram insights...")
            ig_user_id = os.getenv("IG_USER_ID")
            df_ig = fetch_latest_ig_media_insights(ig_user_id, metrics=["impressions", "reach", "engagement"])
            
            if not df_ig.empty:
                logger.info(f"✅ Latest Instagram insights: {len(df_ig)} records")
                print("Latest Instagram insights sample:")
                print(df_ig.head())
            else:
                logger.warning("⚠️ No latest Instagram insights data")
        
        # Test combined organic insights
        logger.info("Testing combined organic insights with 'latest' preset...")
        df_combined = get_organic_insights(date_preset="latest", include_instagram=True)
        
        if not df_combined.empty:
            logger.info(f"✅ Combined organic insights: {len(df_combined)} records")
            print("Combined organic insights sample:")
            print(df_combined.head())
            
            # Show breakdown by source
            if 'source' in df_combined.columns:
                source_counts = df_combined['source'].value_counts()
                logger.info(f"Records by source: {source_counts.to_dict()}")
        else:
            logger.warning("⚠️ No combined organic insights data")
        
        logger.info("✅ Organic insights test completed")
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
    except Exception as e:
        logger.error(f"❌ Test error: {e}", exc_info=True)

if __name__ == "__main__":
    test_organic_insights()
