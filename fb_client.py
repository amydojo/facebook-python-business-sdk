"""
Facebook Business SDK client initialization.
Official docs: https://developers.facebook.com/docs/business-sdk/getting-started/
"""
import os
import logging

# Official Facebook Business SDK imports
# Official docs: https://developers.facebook.com/docs/business-sdk/
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
        if app_id and app_secret:
            self.api = FacebookAdsApi.init(
                app_id=app_id,
                app_secret=app_secret,
                access_token=access_token
            )
            logger.info("✅ Facebook API initialized with app secret proof")
        else:
            self.api = FacebookAdsApi.init(access_token=access_token)
            logger.info("✅ Facebook API initialized with access token only")
        
        # Initialize ad account
        account_id = f"act_{ad_account_id}" if not ad_account_id.startswith("act_") else ad_account_id
        self.account = AdAccount(account_id)
        
        self._initialized = True
        logger.info(f"✅ Facebook SDK initialized for Ad Account: {account_id}")
    
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
            return {
                "success": True,
                "account_name": account_info.get("name"),
                "account_status": account_info.get("account_status"),
                "currency": account_info.get("currency")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

# Initialize global client instance
fb_client = FacebookClient()

if __name__ == "__main__":
    # Test initialization
    import os
    # Set test environment variables (replace with actual values)
    # os.environ["META_ACCESS_TOKEN"] = "<your_access_token>"
    # os.environ["AD_ACCOUNT_ID"] = "<your_ad_account_id>"
    # os.environ["META_APP_ID"] = "<your_app_id>"
    # os.environ["META_APP_SECRET"] = "<your_app_secret>"
    
    print(f"fb_client initialized: {fb_client.is_initialized()}")
    if fb_client.is_initialized():
        test_result = fb_client.test_connection()
        print(f"Connection test: {test_result}")
    print(f"fb_client.account: {getattr(fb_client, 'account', None)}")

class FacebookClient:
    """Facebook Business SDK client wrapper."""

    def __init__(self):
        self.api = None
        self.account = None
        self._initialize()

    def _initialize(self):
        """
        Initialize Facebook Ads API with app secret proof.
        Official docs: https://developers.facebook.com/docs/business-sdk/guides/setup
        """
        # Environment variable validation
        access_token = os.getenv("META_ACCESS_TOKEN")
        app_id = os.getenv("META_APP_ID")
        app_secret = os.getenv("META_APP_SECRET")
        ad_account_id = os.getenv("AD_ACCOUNT_ID")

        # Log environment status (without exposing sensitive values)
        logger.info(f"🔍 Environment check - AD_ACCOUNT_ID: {ad_account_id}, ACCESS_TOKEN set: {bool(access_token)}")
        logger.info(f"🔍 APP_ID set: {bool(app_id)}, APP_SECRET set: {bool(app_secret)}")

        if not SDK_AVAILABLE:
            logger.error("❌ Facebook Business SDK not available due to import errors")
            return

        if not access_token:
            logger.error("❌ Missing META_ACCESS_TOKEN - cannot initialize Facebook API")
            return

        if not ad_account_id:
            logger.error("❌ Missing AD_ACCOUNT_ID - cannot initialize ad account")
            return

        try:
            # Initialize SDK with app secret proof if available
            # Official docs: https://developers.facebook.com/docs/business-sdk/getting-started/
            if app_id and app_secret:
                self.api = FacebookAdsApi.init(
                    app_id=app_id,
                    app_secret=app_secret,
                    access_token=access_token,
                    api_version="v18.0"
                )
                logger.info("✅ FacebookAdsApi initialized with app secret proof")
            else:
                self.api = FacebookAdsApi.init(
                    access_token=access_token,
                    api_version="v18.0"
                )
                logger.info("✅ FacebookAdsApi initialized with access token only")

            # Create AdAccount instance
            # Format: act_{account_id}
            account_id_formatted = f"act_{ad_account_id}" if not ad_account_id.startswith("act_") else ad_account_id
            self.account = AdAccount(account_id_formatted)
            logger.info(f"✅ Facebook SDK initialized for Ad Account: {account_id_formatted}")

        except Exception as e:
            # Log full exception details for debugging
            logger.error(f"❌ Error during Facebook API initialization: {e} (type: {type(e)})", exc_info=True)
            self.api = None
            self.account = None

    def get_account(self):
        """Return the ad account object or None if not available."""
        return self.account

    def is_initialized(self):
        """Check if the client is properly initialized."""
        return self.api is not None and self.account is not None

    def test_connection(self):
        """
        Test the connection by fetching basic account info.
        Returns: dict with connection status and account info
        """
        if not self.is_initialized():
            return {"success": False, "error": "Client not initialized"}

        try:
            # Test connection by getting account info
            account_info = self.account.api_get(fields=['name', 'account_status', 'currency'])
            return {
                "success": True,
                "account_name": account_info.get('name'),
                "account_status": account_info.get('account_status'),
                "currency": account_info.get('currency'),
                "account_id": self.account.get_id()
            }
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e} (type: {type(e)})", exc_info=True)
            return {"success": False, "error": str(e)}

# Global client instance - initialized once at module import
try:
    fb_client = FacebookClient()
except Exception as e:
    logger.error(f"❌ Failed to initialize global fb_client: {e}", exc_info=True)
    fb_client = FacebookClient()  # Create instance even if initialization fails