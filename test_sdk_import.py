
#!/usr/bin/env python3
"""
Test script to verify Facebook Business SDK imports and initialization.
Run this to debug import issues: python test_sdk_import.py

Official docs: https://developers.facebook.com/docs/business-sdk/getting-started/
"""
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test Facebook Business SDK imports."""
    logger.info("🧪 Testing Facebook Business SDK imports...")
    
    try:
        # Test core imports
        from facebook_business.api import FacebookAdsApi
        from facebook_business.adobjects.adaccount import AdAccount
        from facebook_business.exceptions import FacebookRequestError
        
        # Log import locations
        import facebook_business.api as fb_api
        import facebook_business.adobjects.adaccount as fb_adaccount
        
        logger.info("✅ Core imports successful:")
        logger.info(f"  - FacebookAdsApi from: {fb_api.__file__}")
        logger.info(f"  - AdAccount from: {fb_adaccount.__file__}")
        logger.info(f"  - FacebookRequestError: {FacebookRequestError}")
        
        # Test exception class validity
        if issubclass(FacebookRequestError, BaseException):
            logger.info("✅ FacebookRequestError is a valid exception class")
        else:
            logger.error("❌ FacebookRequestError is not a valid exception class")
            return False
            
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def test_environment():
    """Test environment variables."""
    logger.info("🧪 Testing environment variables...")
    
    required_vars = ["META_ACCESS_TOKEN", "AD_ACCOUNT_ID"]
    optional_vars = ["META_APP_ID", "META_APP_SECRET", "PAGE_ID"]
    
    missing_required = []
    missing_optional = []
    
    for var in required_vars:
        if os.getenv(var):
            logger.info(f"✅ {var} is set")
        else:
            logger.error(f"❌ {var} is missing")
            missing_required.append(var)
    
    for var in optional_vars:
        if os.getenv(var):
            logger.info(f"✅ {var} is set")
        else:
            logger.warning(f"⚠️ {var} is missing (optional)")
            missing_optional.append(var)
    
    if missing_required:
        logger.error(f"❌ Missing required environment variables: {missing_required}")
        return False
    
    if missing_optional:
        logger.warning(f"⚠️ Missing optional environment variables: {missing_optional}")
    
    return True

def test_initialization():
    """Test Facebook API initialization."""
    logger.info("🧪 Testing Facebook API initialization...")
    
    try:
        from facebook_business.api import FacebookAdsApi
        from facebook_business.adobjects.adaccount import AdAccount
        
        access_token = os.getenv("META_ACCESS_TOKEN")
        app_id = os.getenv("META_APP_ID")
        app_secret = os.getenv("META_APP_SECRET")
        ad_account_id = os.getenv("AD_ACCOUNT_ID")
        
        if not access_token:
            logger.error("❌ Cannot test initialization: META_ACCESS_TOKEN missing")
            return False
        
        # Initialize API
        if app_id and app_secret:
            api = FacebookAdsApi.init(
                app_id=app_id,
                app_secret=app_secret,
                access_token=access_token,
                api_version="v18.0"
            )
            logger.info("✅ API initialized with app secret proof")
        else:
            api = FacebookAdsApi.init(
                access_token=access_token,
                api_version="v18.0"
            )
            logger.info("✅ API initialized with access token only")
        
        # Test account creation
        if ad_account_id:
            account_id_formatted = f"act_{ad_account_id}" if not ad_account_id.startswith("act_") else ad_account_id
            account = AdAccount(account_id_formatted)
            logger.info(f"✅ AdAccount created: {account_id_formatted}")
            
            # Test basic API call (this might fail if token is invalid, but that's expected)
            try:
                account_info = account.api_get(fields=['name'])
                logger.info(f"✅ API call successful - Account: {account_info.get('name', 'Unknown')}")
            except Exception as e:
                logger.warning(f"⚠️ API call failed (possibly invalid token): {e}")
        else:
            logger.warning("⚠️ Cannot test AdAccount: AD_ACCOUNT_ID missing")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        return False

def test_shadow_check():
    """Check for local facebook_business modules that might shadow the SDK."""
    logger.info("🧪 Checking for shadow modules...")
    
    import facebook_business
    fb_path = facebook_business.__file__
    logger.info(f"facebook_business module loaded from: {fb_path}")
    
    # Check if it's in the current directory (bad)
    current_dir = os.getcwd()
    if fb_path.startswith(current_dir):
        logger.error(f"❌ facebook_business is being loaded from local directory: {fb_path}")
        logger.error("This will shadow the installed SDK. Remove the local facebook_business directory.")
        return False
    else:
        logger.info("✅ facebook_business is loaded from installed package (not local shadow)")
        return True

def main():
    """Run all tests."""
    logger.info("🚀 Starting Facebook Business SDK verification tests")
    logger.info("=" * 50)
    
    tests = [
        ("Shadow Module Check", test_shadow_check),
        ("SDK Imports", test_imports),
        ("Environment Variables", test_environment),
        ("API Initialization", test_initialization),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Facebook Business SDK is ready to use.")
        return 0
    else:
        logger.error("💥 SOME TESTS FAILED! Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
