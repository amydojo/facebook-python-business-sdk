
"""
Facebook Business SDK client initialization.
Official docs: https://developers.facebook.com/docs/business-sdk/getting-started/
"""
import os
import logging

# Official Facebook Business SDK imports
try:
    from facebook_business.api import FacebookAdsApi
    from facebook_business.adobjects.adaccount import AdAccount
    from facebook_business.exceptions import FacebookRequestError
    import facebook_business.api as fb_api_module
    import facebook_business.adobjects.adaccount as fb_adaccount_module
    SDK_AVAILABLE = True
except ImportError as e:
    logging.error(f"❌ Facebook Business SDK import error: {e}", exc_info=True)
    FacebookAdsApi = None
    AdAccount = None
    FacebookRequestError = None
    SDK_AVAILABLE = False

logger = logging.getLogger(__name__)

def validate_environment_vars():
    """
    Check that required environment variables for paid insights are present.
    Returns list of missing variables.
    """
    missing = []
    required_vars = [
        "META_ACCESS_TOKEN",
        "AD_ACCOUNT_ID"
    ]

    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)

    if missing:
        logger.warning(f"Missing environment variables: {', '.join(missing)}")
    else:
        logger.info("All required credentials are set.")
    
    return missing

# Log import paths to confirm correct SDK loading
if SDK_AVAILABLE:
    logger.info(f"✅ facebook_business.api loaded from: {fb_api_module.__file__}")
    logger.info(f"✅ facebook_business.adobjects.adaccount loaded from: {fb_adaccount_module.__file__}")

class FacebookClient:
    """
    Facebook Business SDK client wrapper with robust initialization.
    Official docs: https://developers.facebook.com/docs/business-sdk/getting-started/
    """

    def __init__(self):
        self.api = None
        self.account = None
        self._initialized = False

        if not SDK_AVAILABLE:
            logger.error("❌ Facebook Business SDK not available")
            return

        try:
            self._initialize_api()
        except Exception as e:
            logger.error(f"❌ Failed to initialize Facebook client: {e}", exc_info=True)

    def _initialize_api(self):
        """Initialize Facebook Ads API with environment variables."""
        access_token = os.getenv("META_ACCESS_TOKEN")
        ad_account_id = os.getenv("AD_ACCOUNT_ID")
        app_id = os.getenv("META_APP_ID")
        app_secret = os.getenv("META_APP_SECRET")

        if not access_token or not ad_account_id:
            logger.error("❌ Missing required environment variables: META_ACCESS_TOKEN or AD_ACCOUNT_ID")
            return

        # Initialize API with optional app secret proof for enhanced security
        try:
            if app_id and app_secret:
                self.api = FacebookAdsApi.init(
                    app_id=app_id,
                    app_secret=app_secret,
                    access_token=access_token,
                    api_version="v23.0"
                )
                logger.info("✅ Facebook API initialized with app secret proof")
            else:
                self.api = FacebookAdsApi.init(
                    access_token=access_token,
                    api_version="v23.0"
                )
                logger.info("✅ Facebook API initialized with access token only")

            # Initialize ad account
            account_id = f"act_{ad_account_id}" if not ad_account_id.startswith("act_") else ad_account_id
            self.account = AdAccount(account_id)

            self._initialized = True
            logger.info(f"✅ Facebook SDK initialized for Ad Account: {account_id}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Facebook API: {e}")
            self._initialized = False

    def is_initialized(self):
        """Check if client is properly initialized."""
        return self._initialized and self.api is not None and self.account is not None

    def test_connection(self):
        """Test API connection and return status."""
        if not self.is_initialized():
            return {"success": False, "error": "Client not initialized"}

        try:
            # Simple API call to test connection
            account_info = self.account.api_get(fields=["name", "account_status", "currency"])
            
            # Handle different response types
            if hasattr(account_info, 'export_all_data'):
                account_data = account_info.export_all_data()
            elif isinstance(account_info, dict):
                account_data = account_info
            else:
                account_data = {"name": "Unknown", "account_status": "UNKNOWN", "currency": "USD"}
            
            return {
                "success": True,
                "account_name": account_data.get("name"),
                "account_status": account_data.get("account_status"),
                "currency": account_data.get("currency")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

def validate_credentials():
    """
    Validate Facebook Business SDK credentials and account access.
    
    Returns:
        bool: True if credentials are valid and account is accessible
    """
    try:
        # First check environment variables
        missing_vars = validate_environment_vars()
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            return False
        
        # Check if client is initialized
        if not fb_client.is_initialized():
            logger.error("Facebook client not initialized")
            return False
            
        # Test connection
        test_result = fb_client.test_connection()
        if test_result.get("success"):
            logger.info("✅ Facebook credentials validated successfully")
            return True
        else:
            logger.error(f"❌ Facebook credential validation failed: {test_result.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error validating Facebook credentials: {e}")
        return False

# Initialize global client instance
fb_client = FacebookClient()

if __name__ == "__main__":
    # Test initialization
    print(f"SDK available: {SDK_AVAILABLE}")
    print(f"fb_client initialized: {fb_client.is_initialized()}")
    if fb_client.is_initialized():
        test_result = fb_client.test_connection()
        print(f"Connection test: {test_result}")
        print(f"Credentials valid: {validate_credentials()}")
    else:
        print("fb_client not initialized - check environment variables")
        missing = validate_environment_vars()
        print(f"Missing variables: {missing}")
