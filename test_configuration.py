
#!/usr/bin/env python3
"""
Test script to validate environment variables and API connectivity
"""
import os
import logging
from fb_client import validate_credentials
from fetch_organic import validate_organic_environment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_environment_variables():
    """Test that all required environment variables are set"""
    required_vars = [
        "META_ACCESS_TOKEN",
        "AD_ACCOUNT_ID", 
        "PAGE_ID",
        "IG_USER_ID",
        "PAGE_ACCESS_TOKEN",
        "META_APP_ID",
        "META_APP_SECRET",
        "OPENAI_API_KEY"
    ]
    
    missing = []
    present = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            present.append(f"{var}: {'*' * min(8, len(value))}...")
        else:
            missing.append(var)
    
    logger.info("✅ Environment variables present:")
    for var in present:
        logger.info(f"  {var}")
    
    if missing:
        logger.error("❌ Missing environment variables:")
        for var in missing:
            logger.error(f"  {var}")
        return False
    
    return True

def test_api_connectivity():
    """Test API connectivity"""
    logger.info("🧪 Testing Facebook API connectivity...")
    
    try:
        if validate_credentials():
            logger.info("✅ Facebook API validation successful")
        else:
            logger.error("❌ Facebook API validation failed")
            return False
    except Exception as e:
        logger.error(f"❌ Facebook API test error: {e}")
        return False
    
    logger.info("🧪 Testing organic insights environment...")
    organic_env = validate_organic_environment()
    logger.info(f"📋 Organic environment status: {organic_env}")
    
    return True

if __name__ == "__main__":
    logger.info("🚀 Starting configuration tests...")
    
    env_ok = test_environment_variables()
    if env_ok:
        api_ok = test_api_connectivity()
        if api_ok:
            logger.info("🎉 All tests passed!")
        else:
            logger.error("💥 API connectivity tests failed")
    else:
        logger.error("💥 Environment variable tests failed")
